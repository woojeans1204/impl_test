import os
os.environ['HF_HOME'] = '/scratch/x3326a36/hf_cache' # 적절히 바꾸기
import requests
import json
import tiktoken
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 설정 및 경로
# ==========================================
# 스탠포드 Alpaca 데이터셋 원본 URL
DATA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

# 현재 스크립트 위치(scripts)의 상위 폴더(루트)를 기준으로 data/alpaca 경로 지정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(base_dir, "data", "alpaca")
os.makedirs(save_dir, exist_ok=True)

# ==========================================
# 2. 데이터 다운로드
# ==========================================
json_path = os.path.join(save_dir, "alpaca_data.json")

if not os.path.exists(json_path):
    print(f"Downloading Alpaca dataset to {json_path}...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        with open(json_path, "wb") as f:
            f.write(response.content)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading data: {e}")
        exit(1)
else:
    print("Alpaca JSON file already exists.")

# JSON 로드
with open(json_path, "r", encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} examples.")

# ==========================================
# 3. 프롬프트 포맷팅 함수 (핵심!)
# ==========================================
def format_alpaca(entry):
    # 입력(Input)이 있는 경우와 없는 경우를 구분해서 포맷팅
    if entry.get("input", "") != "":
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{entry['instruction']}

### Input:
{entry['input']}

### Response:
{entry['output']}"""
    else:
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{entry['instruction']}

### Response:
{entry['output']}"""
    
    # [중요] 대화가 끝났음을 알리는 EOS 토큰 추가
    text += "<|endoftext|>" 
    return text

# ==========================================
# 4. 토크나이징
# ==========================================
enc = tiktoken.get_encoding("gpt2")
all_ids = []

print("Tokenizing data...")
for entry in tqdm(data):
    text = format_alpaca(entry)
    # <|endoftext|> 같은 특수 토큰 허용
    ids = enc.encode(text, allowed_special={'<|endoftext|>'})
    all_ids.extend(ids)

print(f"Total tokens: {len(all_ids):,}")

# ==========================================
# 5. 바이너리 저장 (Train 90% / Val 10%)
# ==========================================
all_ids = np.array(all_ids, dtype=np.uint16)
n = len(all_ids)
split_idx = int(n * 0.9)

train_ids = all_ids[:split_idx]
val_ids = all_ids[split_idx:]

train_path = os.path.join(save_dir, 'train.bin')
val_path = os.path.join(save_dir, 'val.bin')

train_ids.tofile(train_path)
val_ids.tofile(val_path)

print(f"\nDone! Files saved at: {save_dir}")
print(f"- train.bin: {len(train_ids):,} tokens")
print(f"- val.bin:   {len(val_ids):,} tokens")