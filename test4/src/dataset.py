import torch
from torch.utils.data import Dataset, DataLoader
import requests
import os

# 데이터셋 소스 관리
DATA_INFO = {
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "tiny_shakespeare.txt"
    },
    "alice": {
        "url": "https://www.gutenberg.org/files/11/11-0.txt", # Project Gutenberg: Alice in Wonderland
        "filename": "alice_in_wonderland.txt"
    }
}

class CharDataset(Dataset):
    def __init__(self, seq_len=64, dataset_name="shakespeare"):
        self.seq_len = seq_len
        
        # 설정된 데이터셋이 없으면 기본값(shakespeare) 사용
        if dataset_name not in DATA_INFO:
            print(f"Warning: Unknown dataset '{dataset_name}'. Defaulting to 'shakespeare'.")
            dataset_name = "shakespeare"
            
        info = DATA_INFO[dataset_name]
        file_path = os.path.join('data', info['filename'])

        if not os.path.exists('data'):
            os.makedirs('data')
        
        # 파일이 없으면 다운로드
        if not os.path.exists(file_path):
            print(f"Downloading {dataset_name} from {info['url']}...")
            try:
                # Gutenberg 등 일부 사이트는 User-Agent 헤더가 필요할 수 있음
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(info['url'], headers=headers)
                response.raise_for_status() # 에러 발생 시 예외 처리
                
                text = response.text
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception as e:
                print(f"Failed to download: {e}")
                raise e
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # 데이터 전처리
        chars = sorted(list(set(text))) # 고유 문자 집합
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        print(f"[{dataset_name}] Data Loaded. Vocab size: {self.vocab_size}, Total len: {len(self.data)}")

    def __len__(self):
        return len(self.data) - self.seq_len + 1
    
    def __getitem__(self, idx):
        return self.data[idx: idx+self.seq_len]
        
def get_dataloader(batch_size, seq_len, dataset_name="shakespeare"):
    dataset = CharDataset(seq_len, dataset_name=dataset_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True), dataset