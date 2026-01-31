import torch
from torch.optim import Adam
from tqdm import tqdm
import os
import yaml
from datetime import datetime
from pathlib import Path
from torchvision.utils import save_image # 이미지 저장을 위해 추가

from .model import DDPMUNet
from .model_simple import SimpleUNet
from .diffusion import GaussianDiffusion
from .dataset import get_dataloader
from .__init__ import *
from accelerate import Accelerator
from .ema import EMA

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------------------------------------
        # 1. 폴더 구조 생성 로직 (날짜 + 실험명)
        # -------------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config['experiment_name']
        self.root_dir = Path("results") / f"{timestamp}_{exp_name}"
        self.ckpt_dir = self.root_dir / "checkpoints"
        self.sample_dir = self.root_dir / "samples"
        self.log_dir = self.root_dir / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = self.config['train'].get('save_interval', 10)

        # 2. config.yaml 박제 (필수)
        with open(self.root_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f">>> Experiment directory created at: {self.root_dir}")
        # -------------------------------------------------------
        
        model_type = config['model'].get('type', 'ddpm') # 기본값 ddpm
        print(f">>> Loading Model Type: {model_type}")

        if model_type == 'ddpm':
            # 정석 모델 (model.py)
            self.model = DDPMUNet(
                dim=config['model']['dim'],
                in_out_ch=config['data']['channels']
            ).to(self.device)
            
        elif model_type == 'simple':
            # 단순 모델 (model_simple.py)
            self.model = SimpleUNet(
                dim=config['model']['dim'], # simple 모델은 dim=64가 기본이었지만 config값 따라가게 설정
                in_out_ch=config['data']['channels']
            ).to(self.device)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.diffusion = GaussianDiffusion(timesteps=config['diffusion']['timesteps'])

        self.dataloader = get_dataloader(batch_size=config['train']['batch_size'], image_size=config['data']['image_size'])
        self.optimizer = Adam(self.model.parameters(), lr=float(config['train']['lr']))

        # 가속기 설정
        self.accelerator = Accelerator(
            mixed_precision='fp16',
            project_dir=self.log_dir,
            log_with="tensorboard"
        )
        # [추가] 1. 트래커 초기화 (실험 시작 알림)
        # config를 넘겨주면 텐서보드 하이퍼파라미터 탭에 자동 기록됩니다.
        if self.accelerator.is_main_process:
            # -----------------------------------------------------------
            # [수정] Config 평탄화 (Flattening)
            # TensorBoard는 중첩된 dict를 못 받으므로, 한 줄로 펴줍니다.
            # 예: config['data']['image_size'] -> flat_config['data_image_size']
            # -----------------------------------------------------------
            flat_config = {}
            for key, val in config.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        flat_config[f"{key}_{k}"] = v
                else:
                    flat_config[key] = val
            
            self.accelerator.init_trackers(
                project_name=config['experiment_name'], 
                config=flat_config
            )


        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)
        
        # ema
        self.ema = EMA(self.accelerator.unwrap_model(self.model), beta=config['train'].get('ema_beta', 0.995))

        # 검증용 노이즈
        self.fixed_noise = torch.randn(16, config['data']['channels'], 
                                     config['data']['image_size'], 
                                     config['data']['image_size']).to(self.device)
    
    def train(self):
        self.model.train()
        # [추가] 글로벌 스텝 관리 (전체 학습 횟수 카운트용)
        global_step = 0

        for epoch in range(1, self.config['train']['epochs']+1):
            pbar = tqdm(self.dataloader,
                        disable=not self.accelerator.is_main_process)
            pbar.set_description(f"Epoch {epoch}")
            current_loss = float('nan')

            for images, _ in pbar:
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()

                loss = self.diffusion.p_losses(self.model, images, t)
                current_loss = loss.item()

                self.optimizer.zero_grad()
                # loss.backward()
                self.accelerator.backward(loss)
                self.optimizer.step()

                global_step += 1
                if global_step % 10 == 0:
                    self.accelerator.log({"train_loss": loss.item()}, step=global_step)

                if self.accelerator.is_main_process:
                    pbar.set_postfix(loss=loss.item())
                    self.ema.update(self.accelerator.unwrap_model(self.model))
           
            # -------------------------------------------------------
            # 주기적 저장 및 샘플링
            # -------------------------------------------------------
            if epoch % self.save_interval == 0:
                if self.accelerator.is_main_process:
                    self.save_checkpoint(epoch, current_loss)
                    self.sample_fixed_images(epoch)

        self.accelerator.end_training()

                  
    def sample_fixed_images(self, epoch):
        # 1. EMA 모델 준비
        ema_model = self.ema.get_model()
        ema_model.eval()
        
        # 2. Raw 모델 준비 (self.model)
        self.model.eval()
        
        with torch.no_grad():
            # 두 모델을 리스트로 묶어서 반복 처리
            # (모델 객체, 파일명 접두어)
            models_to_run = [
                (self.model, "raw"),
                (ema_model, "ema")
            ]
            
            for model, prefix in models_to_run:
                x = self.fixed_noise.clone()
                
                # Diffusion 역과정 진행
                for t in reversed(range(self.diffusion.timesteps)):
                    # 타임스텝 텐서 생성 (배치 사이즈 16 가정)
                    t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
                    x = self.diffusion.p_sample(model, x, t_tensor, t)
                
                # 후처리: [-1, 1] -> [0, 1]
                x = (x.clamp(-1, 1) + 1) / 2
                
                # 각각 다른 파일명으로 저장
                save_path = self.sample_dir / f"sample_{prefix}_epoch_{epoch:04d}.png"
                save_image(x, save_path, nrow=4)
            
        # 학습 모드로 복구 (필요 시)
        self.model.train()
            
    def save_checkpoint(self, epoch, loss):
        """체크포인트와 Config를 함께 저장"""
        state = {
            'epoch': epoch,
            'model_state': self.accelerator.unwrap_model(self.model).state_dict(),
            'ema_model_state': self.ema.get_model().state_dict(), # EMA 상태도 저장
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config, # Config 박제
            'loss': loss
        }
        
        # 파일명: ckpt_epoch_0010.pth
        filename = self.ckpt_dir / f"ckpt_epoch_{epoch:04d}.pth"
        torch.save(state, filename)
        
        # 최신 파일 갱신 (resume용)
        torch.save(state, self.ckpt_dir / "last.pth")
        print(f"Saved checkpoint: {filename}")