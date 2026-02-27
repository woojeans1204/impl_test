import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, data_dir, block_size, split='train'):
        """
        NanoGPT 스타일의 바이너리 데이터 로더
        Args:
            data_dir (str): 'train.bin', 'val.bin' 파일이 있는 폴더 경로
            block_size (int): 모델의 문맥 길이 (Context Length)
            split (str): 'train' 또는 'val'
        """
        super().__init__()
        
        # 파일 경로 확인
        bin_path = os.path.join(data_dir, f'{split}.bin')
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"데이터 파일이 없습니다: {bin_path}\n 먼저 prepare.py를 실행해서 .bin 파일을 만들어주세요.")

        # 1. np.memmap 사용 (핵심!)
        # 파일을 RAM에 다 올리지 않고, 가상 메모리처럼 매핑만 해둡니다.
        # 대용량 데이터도 순식간에 로딩됩니다.
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
        self.block_size = block_size
        
        # 데이터 길이: (전체 토큰 수) - (문맥 길이)
        # 마지막 토큰에서도 block_size만큼 뒤를 봐야 하므로 빼줍니다.
        self.data_size = len(self.data) - block_size

    def __len__(self):
        full_len = len(self.data)
        return full_len // 100

    def __getitem__(self, idx):
        # 2. 데이터 슬라이싱 (입력과 정답 생성)
        # 텍스트: "Hello World"
        # x (입력): "Hello"
        # y (정답): "ello " (한 칸 뒤로 밀린 예측값)
        # [중요] idx를 버리고, 전체 데이터셋(len(self.data)) 범위에서 랜덤 시작점 선택
        # block_size와 상관없이 1000으로 나눴어도, 여기서 전체 범위를 커버함
        random_start = np.random.randint(0, len(self.data) - self.block_size - 1)
        idx = random_start
        # memmap에서 해당 위치의 데이터를 가져옴 (Disk I/O 발생하지만 OS가 캐싱함)
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # numpy -> torch tensor 변환 (int64로 변환해야 임베딩 레이어에 들어감)
        data_tensor = torch.from_numpy(chunk.astype(np.int64))
        
        x = data_tensor[:-1] # 입력
        y = data_tensor[1:]  # 정답 (Next Token)
        
        return x, y

# -----------------------------------------------------------------------------
# DataLoader 생성 헬퍼 함수
# -----------------------------------------------------------------------------
def create_dataloader(data_dir, block_size, batch_size, split='train', num_workers=4):
    """
    V100 학습에 최적화된 DataLoader를 생성해서 반환합니다.
    """
    dataset = GPTDataset(data_dir, block_size, split)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'), # 학습 때는 섞고, 검증 때는 안 섞음
        num_workers=num_workers,    # CPU 병렬 처리로 데이터 로딩 가속
        pin_memory=True,            # GPU 전송 속도 향상 (V100 필수 옵션)
        drop_last=True              # 배치가 딱 안 떨어지면 버림 (Shape 오류 방지)
    )
    
    return loader