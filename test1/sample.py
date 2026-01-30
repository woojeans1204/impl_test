import torch
import yaml
import os
from src.diffusion import GaussianDiffusion
from src.model import SimpleUNet
from torchvision.utils import save_image
from src.__init__ import *

def sample():
    with open(CONFIG_DIR/"mnist.yml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleUNet(dim=64).to(device)
    diffusion = GaussianDiffusion(timesteps=config['diffusion']['timesteps'])

    checkpoint_path = CHECK_DIR/"model_epoch_99.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded {checkpoint_path}")
    model.eval()

    print("Sampling images...")

    with torch.no_grad():
        batch_size = 16
        img_size = 32
        img = torch.randn((batch_size, 1, img_size, img_size), device=device)

        for i in reversed(range(0, diffusion.timesteps)):
            t = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            img = diffusion.p_sample(model, img, t, i)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_image(img, OUTPUT_DIR/"output.png", nrow=4, normalize=True)
        print("Sampling 완성")


if __name__ == '__main__':
    sample()