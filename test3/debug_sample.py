import torch
import yaml
import os
from src.diffusion import GaussianDiffusion
from src.model import SimpleUNet
from torchvision.utils import save_image
from src.__init__ import *

def sample():
    with open(CONFIG_DIR/"config.yml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleUNet(dim=config['model']['dim'], in_out_ch=config['data']['channels']).to(device)
    diffusion = GaussianDiffusion(timesteps=config['diffusion']['timesteps'])

    checkpoint_path = CHECK_DIR/"model_epoch_20.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded {checkpoint_path}")
    model.eval()

    print("Sampling images...")

    with torch.no_grad():
        batch_size = 16
        img_size = config['data']['image_size']
        img = torch.randn((batch_size, 3, img_size, img_size), device=device).float()
        
        for i in reversed(range(0, diffusion.timesteps)):
            t = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            img = diffusion.p_sample(model, img, t, i)

            # [디버깅 코드 1] 중간 단계에서 값이 발산하는지 확인
            if i % 100 == 0:
                print(f"Step {i}: min={img.min().item():.4f}, max={img.max().item():.4f}")

        # [디버깅 코드 2] 최종 값을 강제로 [-1, 1]로 자르기 (발산 방지)
        # img = torch.clamp(img, -1.0, 1.0)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_image(img, OUTPUT_DIR/"output.png", nrow=4, normalize=True, value_range=(-1, 1))
        print("Sampling 완성")


if __name__ == '__main__':
    sample()