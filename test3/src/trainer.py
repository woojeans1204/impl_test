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
        
        # [폴더 경로 결정: 이어하기 vs 새로하기]
        base_exp_name = config['experiment_name']
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = results_dir / base_exp_name
        self.root_dir = None
        
        if target_path.exists() and (target_path / "config.yaml").exists():
            if self._check_config_consistency(target_path / "config.yaml"):
                print(f">>> [Resume] Config matches. Resuming experiment in: {target_path}")
                self.root_dir = target_path
            else:
                print(f">>> [Safety] Config DIFFERS! Creating NEW version.")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.root_dir = results_dir / f"{base_exp_name}_v_{timestamp}"
        else:
            self.root_dir = target_path
            print(f">>> Starting NEW experiment: {self.root_dir}")

        self.ckpt_dir = self.root_dir / "checkpoints"
        self.sample_dir = self.root_dir / "samples"
        self.log_dir = self.root_dir / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = self.config['train'].get('save_interval', 10)

        if not (self.root_dir / "config.yaml").exists():
            with open(self.root_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        
        # --- 모델 초기화 ---
        model_type = config['model'].get('type', 'ddpm')
        if model_type == 'ddpm':
            self.model = DDPMUNet(dim=config['model']['dim'], in_out_ch=config['data']['channels']).to(self.device)
        elif model_type == 'simple':
            self.model = SimpleUNet(dim=config['model']['dim'], in_out_ch=config['data']['channels']).to(self.device)
        else: raise ValueError(f"Unknown model type: {model_type}")
        
        self.diffusion = GaussianDiffusion(timesteps=config['diffusion']['timesteps'])
        self.dataloader = get_dataloader(batch_size=config['train']['batch_size'], image_size=config['data']['image_size'])
        self.optimizer = Adam(self.model.parameters(), lr=float(config['train']['lr']))

        # --- Accelerator ---
        self.accelerator = Accelerator(mixed_precision='fp16', project_dir=self.log_dir, log_with="tensorboard")
        if self.accelerator.is_main_process:
            flat_config = {}
            for key, val in config.items():
                if isinstance(val, dict):
                    for k, v in val.items(): flat_config[f"{key}_{k}"] = v
                else: flat_config[key] = val
            self.accelerator.init_trackers(project_name=base_exp_name, config=flat_config)

        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)
        self.ema = EMA(self.accelerator.unwrap_model(self.model), beta=config['train'].get('ema_beta', 0.995))
        self.fixed_noise = torch.randn(16, config['data']['channels'], config['data']['image_size'], config['data']['image_size']).to(self.device)

        self.start_epoch = 1
        self.load_checkpoint()

    # [수정] 어디가 다른지 범인을 찾아서 출력해주는 함수
    def _check_config_consistency(self, saved_config_path):
        try:
            with open(saved_config_path, 'r') as f:
                saved_conf = yaml.safe_load(f)
            
            # 1. 1차 단순 비교
            if saved_conf == self.config:
                return True
            
            # 2. 다르다면 상세 비교 수행 (디버깅용)
            print(f"\n>>> [Config Mismatch Detected] Diff Report:")
            self._compare_dicts(saved_conf, self.config)
            print("------------------------------------------\n")
            
            return False
        except Exception as e:
            print(f">>> Config check failed: {e}")
            return False

    # [추가] 재귀적으로 딕셔너리 비교하여 다른 점 출력
    def _compare_dicts(self, d1, d2, path=""):
        for k in d1:
            if k not in d2:
                print(f"   • Missing Key in New Config: {path}{k}")
            else:
                if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                    self._compare_dicts(d1[k], d2[k], path=f"{path}{k}.")
                else:
                    # 값은 같은데 타입만 다른 경우 (예: "1e-4" vs 1e-4) 처리
                    val1, val2 = d1[k], d2[k]
                    if val1 != val2:
                        # 숫자 <-> 문자열 변환 후 같다면 통과시켜주기 (유연한 비교)
                        try:
                            if float(val1) == float(val2):
                                continue # 숫자로 바꾸니 같으면 Pass
                        except:
                            pass
                        
                        print(f"   • Mismatch at [{path}{k}]:")
                        print(f"     - Saved  : {val1} (type: {type(val1).__name__})")
                        print(f"     - Current: {val2} (type: {type(val2).__name__})")

    def load_checkpoint(self):
        resume_path = self.ckpt_dir / "last.pth"
        if not resume_path.exists(): return
        
        print(f">>> Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device)
        
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state'])
        if 'ema_model_state' in checkpoint: 
            self.ema.get_model().load_state_dict(checkpoint['ema_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        print(f">>> Resume successful. Next epoch: {self.start_epoch}")

    def train(self):
        self.model.train()
        global_step = (self.start_epoch - 1) * len(self.dataloader)

        for epoch in range(self.start_epoch, self.config['train']['epochs']+1):
            pbar = tqdm(self.dataloader, disable=not self.accelerator.is_main_process)
            pbar.set_description(f"Epoch {epoch}")
            current_loss = float('nan')

            for images, _ in pbar:
                images = images.to(self.device)
                batch_size = images.shape[0]
                t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()

                loss = self.diffusion.p_losses(self.model, images, t)
                current_loss = loss.item()

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                global_step += 1
                if global_step % 10 == 0:
                    self.accelerator.log({"train_loss": loss.item()}, step=global_step)

                if self.accelerator.is_main_process:
                    pbar.set_postfix(loss=loss.item())
                    self.ema.update(self.accelerator.unwrap_model(self.model))
           
            if epoch % self.save_interval == 0:
                if self.accelerator.is_main_process:
                    self.save_checkpoint(epoch, current_loss)
                    self.sample_fixed_images(epoch)

        self.accelerator.end_training()

    def sample_fixed_images(self, epoch):
        ema_model = self.ema.get_model()
        ema_model.eval()
        self.model.eval()
        with torch.no_grad():
            models_to_run = [(self.model, "raw"), (ema_model, "ema")]
            for model, prefix in models_to_run:
                x = self.fixed_noise.clone()
                for t in reversed(range(self.diffusion.timesteps)):
                    t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
                    x = self.diffusion.p_sample(model, x, t_tensor, t)
                x = (x.clamp(-1, 1) + 1) / 2
                save_path = self.sample_dir / f"sample_{prefix}_epoch_{epoch:04d}.png"
                save_image(x, save_path, nrow=4)
        self.model.train()
            
    def save_checkpoint(self, epoch, loss):
        state = {
            'epoch': epoch,
            'model_state': self.accelerator.unwrap_model(self.model).state_dict(),
            'ema_model_state': self.ema.get_model().state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'loss': loss
        }
        
        # 1. 현재 체크포인트 저장
        filename = self.ckpt_dir / f"ckpt_epoch_{epoch:04d}.pth"
        torch.save(state, filename)
        
        # 2. Resume용 파일(last.pth) 갱신
        torch.save(state, self.ckpt_dir / "last.pth")
        print(f"Saved checkpoint: {filename}")
        
        # ========================================================
        # [추가] 오래된 체크포인트 자동 삭제 로직
        # last.pth는 건드리지 않고, ckpt_epoch_*.pth 파일들만 관리
        # ========================================================
        try:
            # 이름순 정렬 (0010, 0020... 순서이므로 sort하면 시간순 정렬됨)
            all_ckpts = sorted(self.ckpt_dir.glob("ckpt_epoch_*.pth"))
            
            # 남길 개수 (최근 3개)
            keep_num = 3 
            
            if len(all_ckpts) > keep_num:
                # 삭제 대상: 전체 목록 중 '뒤에서 3개'를 뺀 나머지 전부
                ckpts_to_delete = all_ckpts[:-keep_num]
                
                for ckpt in ckpts_to_delete:
                    ckpt.unlink() # 파일 삭제
                    print(f"Deleted old checkpoint: {ckpt.name}")
        except Exception as e:
            print(f"Failed to delete old checkpoints: {e}")