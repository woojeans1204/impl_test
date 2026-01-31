import torch
import matplotlib.pyplot as plt
from src.dataset import get_dataloader # 작성하신 모듈 경로에 맞게 수정
from src.__init__ import *

def check_data():
    # 배치 사이즈 1로 설정하여 데이터 하나만 가져옴
    dataloader = get_dataloader(batch_size=1, image_size=32)
    
    # 데이터 하나 꺼내기 (iter, next 사용)
    images, labels = next(iter(dataloader))
    img_tensor = images[0] # 첫 번째 이미지
    
    # 1. 수치 값 확인 (가장 중요!)
    print(f"Tensor Shape: {img_tensor.shape}")
    print(f"Min Value: {img_tensor.min().item():.4f}") # -1.0 근처여야 함
    print(f"Max Value: {img_tensor.max().item():.4f}") # 1.0 근처여야 함
    print(f"Mean Value: {img_tensor.mean().item():.4f}")
    
    # 2. 이미지로 복원해서 눈으로 보기
    # [-1, 1] -> [0, 1] 로 복원
    img_vis = (img_tensor + 1) / 2
    img_vis = torch.clamp(img_vis, 0, 1) # 안전장치
    
    # [C, H, W] -> [H, W, C] (matplotlib용)
    img_vis = img_vis.permute(1, 2, 0).cpu().numpy()
    
    plt.imshow(img_vis)
    plt.title("Preprocessed Image Check")
    plt.savefig(OUTPUT_DIR/"data.png")


if __name__ == "__main__":
    check_data()