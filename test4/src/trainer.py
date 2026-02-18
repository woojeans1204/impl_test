import torch
import torch.nn.functional as F
from torch.optim import Adam
from accelerate import Accelerator
from tqdm import tqdm
import yaml
import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# 내부 모듈 import
from .model import TextDiffusionTransformer
from .diffusion import GaussianDiffusion
from .dataset import get_dataloader
from .__init__ import *

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 1. 경로 설정 + Accelerator 초기화 + 폴더 생성을 한방에 처리
        self._setup_experiment_and_accelerator()

        # 2. 데이터 로드
        self.dataloader, self.dataset = get_dataloader(
            batch_size=config['train']['batch_size'], 
            seq_len=config['model']['seq_len'],
            dataset_name=config['data'].get('dataset', 'shakespeare')
        )
        
        # 3. 모델 및 옵티마이저 초기화
        self._init_model_and_optimizer()

        # 4. Accelerator Prepare (분산 학습 준비)
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        
        # 5. 체크포인트 로드 (Resume)
        self.start_epoch = 1
        self._load_checkpoint()

    def _setup_experiment_and_accelerator(self):
        """경로 계산 -> Accelerator 초기화 -> 폴더 생성 (순서 중요!)"""
        base_exp_name = self.config['experiment_name']
        results_dir = Path("results")
        
        # --- A. 경로 변수 먼저 계산 (mkdir 금지, Accelerator 금지) ---
        target_path = results_dir / base_exp_name
        self.root_dir = target_path
        self.is_resume = False
        
        # Resume 여부 판단 (경로 존재 여부만 체크)
        if target_path.exists() and (target_path / "config.yaml").exists():
            if self._check_config_consistency(target_path / "config.yaml"):
                self.is_resume = True
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.root_dir = results_dir / f"{base_exp_name}_v_{timestamp}"
        
        # 하위 경로 변수 설정
        self.ckpt_dir = self.root_dir / "checkpoints"
        self.sample_dir = self.root_dir / "samples"
        self.log_dir = self.root_dir / "logs"
        self.save_interval = self.config['train'].get('save_interval', 10)

        # --- B. 경로가 잡혔으니 Accelerator 생성 ---
        self.accelerator = Accelerator(
            mixed_precision='no', 
            project_dir=self.log_dir, # 위에서 잡은 log_dir 사용
            log_with="tensorboard"
        )
        self.device = self.accelerator.device

        # 트래커 초기화 (메인 프로세스만)
        if self.accelerator.is_main_process:
            flat_config = {}
            for key, val in self.config.items():
                if isinstance(val, dict):
                    for k, v in val.items(): flat_config[f"{key}_{k}"] = v
                else: flat_config[key] = val
            
            self.accelerator.init_trackers(
                project_name=self.config['experiment_name'], 
                config=flat_config
            )

        # --- C. 이제 Accelerator가 있으니 메인 프로세스 체크 후 폴더 생성 ---
        if self.accelerator.is_main_process:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.sample_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # 로그 출력
            if self.is_resume:
                print(f">>> [Resume] Config matches. Resuming experiment in: {self.root_dir}")
            else:
                print(f">>> Starting NEW experiment: {self.root_dir}")
            
            # Config 저장
            if not (self.root_dir / "config.yaml").exists():
                with open(self.root_dir / "config.yaml", "w") as f:
                    yaml.dump(self.config, f, default_flow_style=False)

    def _init_model_and_optimizer(self):
        """모델 및 Diffusion, 옵티마이저 초기화"""
        self.model = TextDiffusionTransformer(
            vocab_size=self.dataset.vocab_size,
            seq_len=self.config['model']['seq_len'],
            dim=self.config['model']['dim'],
            depth=self.config['model']['depth'],
            heads=self.config['model']['heads']
        ).to(self.device)
        
        self.diffusion = GaussianDiffusion(timesteps=self.config['diffusion']['timesteps'])
        self.optimizer = Adam(self.model.parameters(), lr=float(self.config['train']['lr']))

    def _load_checkpoint(self):
        """가장 최근 체크포인트 로드"""
        resume_path = self.ckpt_dir / "last.pth"
        if not resume_path.exists():
            return
        
        if self.accelerator.is_main_process:
            print(f">>> Resuming from: {resume_path}")
        
        checkpoint = torch.load(resume_path, map_location=self.device)
        
        # 모델 Unwrapping 후 로드
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        
        if self.accelerator.is_main_process:
            print(f">>> Resume successful. Next epoch: {self.start_epoch}")

    def train(self):
        """메인 학습 루프"""
        self.model.train()
        global_step = (self.start_epoch - 1) * len(self.dataloader)
        epochs = self.config['train']['epochs']

        for epoch in range(self.start_epoch, epochs + 1):
            pbar = tqdm(self.dataloader, disable=not self.accelerator.is_main_process)
            pbar.set_description(f"Epoch {epoch}")
            current_loss = float('nan')

            for indices in pbar:
                # 1. Forward Process
                # token_emb 접근을 위해 unwrap (모델이 DDP 등으로 감싸져 있을 수 있음)
                raw_model = self.accelerator.unwrap_model(self.model)
                
                # Ground Truth: 정수 ID -> 임베딩 벡터
                x_start = raw_model.get_embeds(indices) 
                
                batch_size = x_start.shape[0]
                t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
                
                # 노이즈 주입
                noise = torch.randn_like(x_start)
                x_noisy = self.diffusion.q_sample(x_start, t, noise)

                # 2. Reverse Process Prediction
                predicted_noise = self.model(x_noisy, t)
                
                sqrt_alpha_bar = self.diffusion._extract(self.diffusion.sqrt_alphas_cumprod, t, x_noisy.shape)
                sqrt_one_minus_alpha_bar = self.diffusion._extract(self.diffusion.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
                predicted_x_start = (x_noisy - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar

                # 3. Optimization
                loss_mse = F.mse_loss(predicted_noise, noise)
                raw_model = self.accelerator.unwrap_model(self.model)

                logits = torch.matmul(predicted_x_start, raw_model.token_emb.weight.T) # (B, L, V)
                
                loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), indices.view(-1))

                loss = loss_mse + 50*loss_ce
                current_loss = loss.item()

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                # 4. Logging
                global_step += 1
                if global_step % 10 == 0:
                    self.accelerator.log({"train_loss": loss.item()}, step=global_step)

                if self.accelerator.is_main_process:
                    pbar.set_postfix(loss=loss.item())
           
            # 5. Save & Sample
            if epoch % self.save_interval == 0:
                if self.accelerator.is_main_process:
                    self.save_checkpoint(epoch, current_loss)
                    self.sample(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sample(self, epoch, num_samples=5):
        '''텍스트 생성 샘플링'''
        self.model.eval()
        raw_model = self.accelerator.unwrap_model(self.model)
        
        # 1. Random Noise 생성
        shape = (num_samples, self.config['model']['seq_len'], self.config['model']['dim'])
        x = torch.randn(shape, device=self.device)
        
        # 2. Reverse Diffusion
        for t in reversed(range(0, self.diffusion.timesteps)):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            x = self.diffusion.p_sample(self.model, x, t_tensor, t)

        # 3. Rounding (가장 가까운 토큰 인덱스 찾기)
        emb_weights = raw_model.token_emb.weight
        dist = torch.cdist(x, emb_weights.unsqueeze(0)) # (N, Seq, Vocab)
        pred_ids = dist.argmin(dim=-1) # (N, Seq) 텐서
        
        # 4. Decoding & Saving
        save_path = self.sample_dir / f"sample_epoch_{epoch:04d}.txt"
        
        with open(save_path, "w", encoding='utf-8') as f:
            f.write(f"Epoch {epoch:04d} Generated Samples:\n")
            f.write("=" * 50 + "\n")

            # 배치의 각 샘플(i)에 대해 루프를 돕니다
            for i in range(num_samples):
                # i번째 문장의 토큰 ID 리스트를 가져옵니다
                tokens = pred_ids[i].tolist() 
                # 각 숫자 ID를 문자로 바꿉니다
                decoded_text = "".join([self.dataset.itos[idx] for idx in tokens])
                
                f.write(f"[{i+1}] {decoded_text}\n")
                
                # 첫 번째 샘플만 터미널에 출력
                if i == 0 and self.accelerator.is_main_process:
                    print(f"\n>>> [Epoch {epoch:04d} Sample]: {decoded_text[:60]}...")
                    
        self.model.train()

    def save_checkpoint(self, epoch, loss):
        """체크포인트 저장"""
        state = {
            'epoch': epoch,
            'model_state': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'loss': loss
        }
        
        # 개별 Epoch 저장
        filename = self.ckpt_dir / f"ckpt_epoch_{epoch:04d}.pth"
        torch.save(state, filename)
        
        # Resume용 'last.pth' 갱신
        torch.save(state, self.ckpt_dir / "last.pth")
        print(f"Saved checkpoint: {filename}")
        
        # 오래된 파일 정리
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self, keep_num=3):
        """오래된 체크포인트 삭제 (최근 N개 유지)"""
        try:
            # glob으로 찾아서 정렬 (0010, 0020 순이므로 문자열 정렬 OK)
            all_ckpts = sorted(self.ckpt_dir.glob("ckpt_epoch_*.pth"))
            
            if len(all_ckpts) > keep_num:
                # 삭제 대상: 뒤에서부터 N개를 제외한 앞부분
                for ckpt in all_ckpts[:-keep_num]:
                    ckpt.unlink()
                    print(f"Deleted old checkpoint: {ckpt.name}")
        except Exception as e:
            print(f"Failed to delete old checkpoints: {e}")

    # --- Config Consistency Utils ---
    def _check_config_consistency(self, saved_config_path):
        """저장된 Config와 현재 Config 비교"""
        try:
            with open(saved_config_path, 'r') as f:
                saved_conf = yaml.safe_load(f)
            
            saved_copy = copy.deepcopy(saved_conf)
            current_copy = copy.deepcopy(self.config)

            # 비교 제외 항목 (Training Loop 관련)
            for conf in [saved_copy, current_copy]:
                if 'train' in conf:
                    conf['train'].pop('epochs', None)
                    conf['train'].pop('save_interval', None)

            if saved_copy == current_copy:
                return True
            
            print(f"\n>>> [Config Mismatch Detected] Diff Report:")
            self._compare_dicts(saved_copy, current_copy)
            print("------------------------------------------\n")
            return False
        except Exception as e:
            print(f">>> Config check failed: {e}")
            return False

    def _compare_dicts(self, d1, d2, path=""):
        """딕셔너리 재귀 비교 및 리포팅"""
        for k in d1:
            if k not in d2:
                print(f"   • Missing Key in New Config: {path}{k}")
            else:
                if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                    self._compare_dicts(d1[k], d2[k], path=f"{path}{k}.")
                else:
                    val1, val2 = d1[k], d2[k]
                    if val1 != val2:
                        # 숫자 타입의 미세한 차이(1e-4 vs 0.0001)는 무시
                        try:
                            if float(val1) == float(val2): continue
                        except: pass
                        print(f"   • Mismatch at [{path}{k}]: {val1} vs {val2}")