import torch
import tiktoken
from src.model import GPT
import yaml
from pathlib import Path
# 딕셔너리를 객체처럼 (config.vocab_size) 쓸 수 있게 해주는 클래스
class Config:
    def __init__(self, entries):
        self.__dict__.update(entries)

def generate_text(config_path, checkpoint_path, prompt, max_new_tokens=200, temperature=0.7):
    # 1. 설정 로드
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # 딕셔너리를 객체로 변환 (AttributeError 해결)
    model_config = Config(full_config['model'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. 모델 초기화 및 체크포인트 로드
    model = GPT(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f">>> Model loaded from {checkpoint_path}")

    # 3. 토크나이저 설정
    enc = tiktoken.get_encoding("gpt2")
    
    if prompt == "":
        input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # 4. 텍스트 생성
    print(f">>> Generating based on prompt: {prompt}")
    print("-" * 50)
    
    with torch.no_grad():
        y_gen = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        generated_text = enc.decode(y_gen[0].tolist())
        print(generated_text)
    
    print("-" * 50)

if __name__ == "__main__":
    # 경로 확인 필수!
    # RESULT_PATH = "results/tinystories_gpt_v1"
    RESULT_PATH = Path("results/fineweb_gpt_v2")
    CONFIG_PATH = RESULT_PATH / "config.yaml"
    CHECKPOINT_PATH = RESULT_PATH / "checkpoints/last.pth"
    
    user_prompt = "how to be good at football?"
    # user_prompt = input()
    
    generate_text(CONFIG_PATH, CHECKPOINT_PATH, user_prompt)