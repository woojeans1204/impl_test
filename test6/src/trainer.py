import torch
import os
from pathlib import Path
from accelerate import Accelerator
from tqdm import tqdm
import yaml
import pickle
import tiktoken

class Trainer:
    # [수정] train.py에서 넘겨주는 4가지 재료를 받도록 수정
    def __init__(self, model, train_loader, val_loader, config):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        meta_path = os.path.join(config['data']['data_dir'], 'meta.pkl')
        if os.path.exists(meta_path):
            # 1. 예전처럼 pkl이 있으면 pkl 사용 (셰익스피어 등)
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.itos = meta['itos']
            self.enc = None
            print(f">>> Loaded manual vocab from {meta_path}")
        else:
            # 2. pkl이 없으면 GPT-2 표준 토크나이저 사용 (TinyStories 등)
            self.itos = None
            self.enc = tiktoken.get_encoding("gpt2")
            print(">>> pkl not found. Using GPT-2 encoding (tiktoken) instead.")

        # 1. 경로 설정 및 Accelerator 초기화
        self._setup_experiment_and_accelerator()

        # 2. 하드웨어 설정 (Accelerator가 알아서 device 배분)
        self.model.to(self.device)

        # 3. Optimizer 설정 (모델 안에 정의된 방식 사용)
        # model.py에 configure_optimizers가 있다고 가정
        if hasattr(self.model, 'configure_optimizers'):
            self.optimizer = self.model.configure_optimizers(
                weight_decay=float(self.config['train']['weight_decay']),
                learning_rate=float(self.config['train']['learning_rate']),
                betas=(self.config['train']['beta1'], self.config['train']['beta2']),
                device_type=self.device.type
            )
        else:
            # 없으면 기본 설정
            from torch.optim import AdamW
            self.optimizer = AdamW(self.model.parameters(), lr=float(self.config['train']['learning_rate']))

        # 4. Accelerator Prepare (분산 학습 준비)
        # 중요: 모델, 옵티마이저, 데이터로더를 감싸줘야 함
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        
        # 5. 체크포인트 로드 (Resume)
        self.start_epoch = 1
        self._load_checkpoint()

    def _setup_experiment_and_accelerator(self):
        """경로 계산 -> Accelerator 초기화 -> 폴더 생성"""
        # config 구조에 따라 system이 없을 수도 있으므로 안전하게 get 사용
        if 'system' in self.config:
            base_exp_name = self.config['system'].get('experiment_name', 'default_exp')
        else:
            base_exp_name = self.config.get('experiment_name', 'default_exp')

        results_dir = Path("results")
        self.root_dir = results_dir / base_exp_name
        self.is_resume = False
        
        # Resume 체크
        if self.root_dir.exists() and (self.root_dir / "config.yaml").exists():
            self.is_resume = True
        
        # 하위 폴더 경로
        self.ckpt_dir = self.root_dir / "checkpoints"
        self.sample_dir = self.root_dir / "samples"
        self.log_dir = self.root_dir / "logs"
        self.save_interval = self.config['train'].get('save_interval', 1)

        # Accelerator 생성 (V100용 fp16)
        self.accelerator = Accelerator(
            mixed_precision='fp16', 
            project_dir=self.log_dir,
            log_with="tensorboard"
        )
        self.device = self.accelerator.device

        # 메인 프로세스에서만 폴더 생성 및 로깅 초기화
        if self.accelerator.is_main_process:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.sample_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Config 파일 저장 (새로운 실험일 경우)
            if not (self.root_dir / "config.yaml").exists():
                with open(self.root_dir / "config.yaml", "w") as f:
                    # Namespace 객체라면 dict로 변환 필요, dict면 그대로
                    conf_to_save = self.config
                    if hasattr(self.config, '__dict__'):
                        conf_to_save = vars(self.config)
                    yaml.dump(conf_to_save, f, default_flow_style=False)
            
            # Tracker 초기화
            self.accelerator.init_trackers(project_name=base_exp_name)

    def _load_checkpoint(self):
        """체크포인트 로드 (보안 에러 및 옵티마이저 빈 값 처리 완료)"""
        resume_path = self.ckpt_dir / "last.pth"
        if not resume_path.exists():
            return
        
        if self.accelerator.is_main_process:
            print(f">>> Resuming from: {resume_path}")
        
        # 1. 보안 옵션 해제 (weights_only=False)
        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
        
        # 2. 모델 가중치 로드
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state'])
        
        # 3. [핵심 수정] 옵티마이저 상태가 '진짜로 있을 때만' 로드
        # 우리가 만든 베이스 모델은 optimizer_state가 {} 비어있으므로, 이 부분을 건너뛰어야 함
        opt_state = checkpoint.get('optimizer_state')
        if opt_state and len(opt_state) > 0:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception as e:
                print(f">>> [Warning] Optimizer load failed (ignoring): {e}")
        else:
            if self.accelerator.is_main_process:
                print(">>> [Info] Optimizer state is empty. Starting with a fresh optimizer.")
            
        # 4. 에포크 설정
        self.start_epoch = checkpoint['epoch'] + 1
        
        if self.accelerator.is_main_process:
            print(f">>> Resume successful. Next epoch: {self.start_epoch}")

    def train(self):
        """학습 루프"""
        epochs = self.config['train'].get('epochs', 10)
        global_step = (self.start_epoch - 1) * len(self.train_loader)

        for epoch in range(self.start_epoch, epochs + 1):
            # 학습 모드
            self.model.train()
            pbar = tqdm(self.train_loader, disable=not self.accelerator.is_main_process)
            pbar.set_description(f"Epoch {epoch}")
            
            for x, y in pbar:
                # Forward (GPT 모델 내부에서 Loss 계산)
                # Accelerator가 device 관리를 해주지만 명시적으로 보낼 수도 있음
                _, loss = self.model(x, y)

                # Backward
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                
                # Gradient Clipping
                if self.config['train'].get('grad_clip', 0.0) > 0.0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config['train']['grad_clip'])
                
                self.optimizer.step()

                # Logging
                global_step += 1
                if global_step % 10 == 0:
                    self.accelerator.log({"train_loss": loss.item()}, step=global_step)
                    if self.accelerator.is_main_process:
                        pbar.set_postfix(loss=loss.item())
            
            # Epoch 끝날 때마다 저장 및 샘플링
            if epoch % self.save_interval == 0:
                if self.accelerator.is_main_process:
                    self.save_checkpoint(epoch, loss.item())
                    self.sample(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sample(self, epoch):
        self.model.eval()
        raw_model = self.accelerator.unwrap_model(self.model)
        
        # [수정] 모델의 성장 과정을 비교하기 위한 '고정 프롬프트' 4종
        prompts = [
            "Artificial Intelligence is",          # 1. 정의 및 설명 능력
            "The capital of France is Paris, and", # 2. 지식 확장 능력
            "Once upon a time,",                   # 3. 스토리텔링 능력
            "1 + 1 = 2, 2 + 2 ="                   # 4. 기초 논리/수학 능력
        ]

        save_path = self.sample_dir / f"sample_epoch_{epoch:04d}.txt"
        
        with open(save_path, "w", encoding='utf-8') as f:
            f.write(f"Epoch {epoch:04d} Sampling Result\n")
            f.write("=" * 50 + "\n\n")

            for prompt in prompts:
                # 1. 프롬프트 인코딩
                if self.enc:
                    # Tiktoken (FineWeb, Chat) 방식
                    start_ids = self.enc.encode(prompt)
                else:
                    # pkl (Shakespeare) 방식: 
                    # __init__에 stoi가 없으므로 안전하게 0번 토큰으로 시작
                    start_ids = [0] 
                
                # 텐서 변환 및 GPU 할당
                x = torch.tensor(start_ids, dtype=torch.long, device=self.device).unsqueeze(0)

                # 2. 시드 고정 (중요!)
                # 매번 똑같은 난수 조건에서 생성해야 순수 모델 성능 변화를 비교 가능
                if self.device.type == 'cuda':
                    torch.cuda.manual_seed(42)
                torch.manual_seed(42)
                
                # 3. 텍스트 생성 (Temperature 0.7로 설정하여 너무 엉뚱한 말 방지)
                y_gen = raw_model.generate(x, max_new_tokens=100, temperature=0.7)
                
                # 4. 디코딩 (숫자 -> 텍스트)
                token_ids = y_gen[0].tolist()
                if self.itos:
                    generated_text = "".join([self.itos[i] for i in token_ids])
                else:
                    generated_text = self.enc.decode(token_ids)
                
                # 5. 결과 파일 기록
                # tiktoken 사용 시 프롬프트 텍스트를 제외한 뒷부분만 깔끔하게 저장
                if self.enc:
                    output_text = generated_text[len(prompt):]
                else:
                    output_text = generated_text

                f.write(f"[Prompt]: {prompt}\n")
                f.write(f"[Output]: {output_text}\n")
                f.write("-" * 30 + "\n\n")
            
        if self.accelerator.is_main_process:
            print(f"\n>>> [Epoch {epoch:04d}] Fixed samples saved to {save_path}")
            
        self.model.train()

    def save_checkpoint(self, epoch, loss):
        """체크포인트 저장"""
        # Config 객체가 Namespace일 수도, dict일 수도 있으므로 처리
        conf_to_save = self.config
        if hasattr(self.config, '__dict__'):
            conf_to_save = vars(self.config)

        state = {
            'epoch': epoch,
            'model_state': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': conf_to_save,
            'loss': loss
        }
        
        filename = self.ckpt_dir / f"ckpt_epoch_{epoch:04d}.pth"
        torch.save(state, filename)
        torch.save(state, self.ckpt_dir / "last.pth")
        
        # 오래된 파일 정리
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self, keep_num=3):
        try:
            all_ckpts = sorted(self.ckpt_dir.glob("ckpt_epoch_*.pth"))
            if len(all_ckpts) > keep_num:
                for ckpt in all_ckpts[:-keep_num]:
                    ckpt.unlink()
        except Exception as e:
            print(f"Cleanup failed: {e}")