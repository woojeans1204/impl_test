import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.__init__ import DATA_DIR

def get_dataloader(batch_size=128):
    # [0, 1] -> [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), # 타입 변환, 값 범위 0~255->[0, 1]
        transforms.Lambda(lambda t: (t*2) - 1)
    ])

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)