import torch
from torch.optim import Adam
from tqdm import tqdm
import os

from .model import SimpleUNet
from .diffusion import GaussianDiffusion
from .dataset import get_dataloader
from .__init__ import *
from accelerate import Accelerator
from .ema import EMA

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SimpleUNet(dim=config['model']['dim'],
                                in_out_ch=config['data']['channels']).to(self.device)
        self.diffusion = GaussianDiffusion(timesteps=config['diffusion']['timesteps'])

        self.dataloader = get_dataloader(batch_size=config['train']['batch_size'], image_size=config['data']['image_size'])
        self.optimizer = Adam(self.model.parameters(), lr=float(config['train']['lr']))

        os.makedirs('checkpoints', exist_ok=True)

        # 가속기 설정
        self.accelerator = Accelerator(mixed_precision='fp16')
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)
        
        # ema
        self.ema = EMA(self.accelerator.unwrap_model(self.model), beta=0.9999)

    
    def train(self):
        self.model.train()
        for epoch in range(self.config['train']['epochs']):
            pbar = tqdm(self.dataloader,
                        disable=not self.accelerator.is_main_process)
            pbar.set_description(f"Epoch {epoch}")

            for images, _ in pbar:
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()

                loss = self.diffusion.p_losses(self.model, images, t)

                self.optimizer.zero_grad()
                # loss.backward()
                self.accelerator.backward(loss)
                self.optimizer.step()

                if self.accelerator.is_main_process:
                    pbar.set_postfix(loss=loss.item())
                    self.ema.update(self.accelerator.unwrap_model(self.model))

            if (epoch+1) % 10 == 0:
                if self.accelerator.is_main_process:
                    state_dict = self.ema.get_model().state_dict()
                    torch.save(self.ema.get_model().state_dict(), CHECK_DIR / f"model_epoch_{epoch+1}.pth")