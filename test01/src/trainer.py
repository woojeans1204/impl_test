import torch
from torch.optim import Adam
from tqdm import tqdm
import os

from .model import SimpleUNet
from .diffusion import GaussianDiffusion
from .dataset import get_dataloader
from .__init__ import CHECK_DIR

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SimpleUNet(dim=config['model']['dim']).to(self.device)
        self.diffusion = GaussianDiffusion(timesteps=config['diffusion']['timesteps'])

        self.dataloader = get_dataloader(batch_size=config['train']['batch_size'])
        self.optimizer = Adam(self.model.parameters(), lr=float(config['train']['lr']))

        os.makedirs('checkpoints', exist_ok=True)
    
    def train(self):
        self.model.train()
        for epoch in range(self.config['train']['epochs']):
            pbar = tqdm(self.dataloader)
            pbar.set_description(f"Epoch {epoch}")

            for images, _ in pbar:
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()

                loss = self.diffusion.p_losses(self.model, images, t)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(loss=loss.item())
                
            if (epoch+1) % 10 == 0:
                torch.save(self.model.state_dict(), CHECK_DIR / f"model_epoch_{epoch}.pth")