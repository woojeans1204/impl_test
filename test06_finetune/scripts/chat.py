import torch
import tiktoken
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts 폴더
parent_dir = os.path.dirname(current_dir)               # 상위 폴더 (test6)
sys.path.append(parent_dir)
# src 폴더 인식을 위해 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import GPT, GPTConfig 

# ==========================================
# 1. 설정
# ==========================================
CHECKPOINT_PATH = "../results/gpt2_large_alpaca_finetune/checkpoints/last.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. 모델 로드 및 설정 변환 (핵심!)
# ==========================================
print(f">>> Loading model from {CHECKPOINT_PATH}...")
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: 체크포인트 파일이 없습니다 ({CHECKPOINT_PATH})")
    exit()

# weights_only=False로 로드 (커스텀 클래스/딕셔너리 호환)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
raw_config = checkpoint['config']

# [Config 변환 로직] 딕셔너리를 GPTConfig 객체로 안전하게 변환
if isinstance(raw_config, dict):
    # 만약 전체 YAML 설정(system, train, model 등)이 들어있다면 'model'만 추출
    if 'model' in raw_config:
        model_args = raw_config['model']
    else:
        model_args = raw_config
    
    # 딕셔너리를 풀어서 객체 생성
    config = GPTConfig(**model_args)
else:
    # 이미 객체라면 그대로 사용
    config = raw_config

print(f">>> Model Config Loaded: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")

# 모델 초기화
model = GPT(config)
model.load_state_dict(checkpoint['model_state'])
model.to(DEVICE)
model.eval()

# ==========================================
# 3. 인퍼런스 함수
# ==========================================
enc = tiktoken.get_encoding("gpt2")

def generate_response(user_input):
    # Alpaca 포맷 적용
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_input}

### Response:
"""
    # 입력 인코딩
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        # max_new_tokens를 넉넉히 줌 (Stop Token으로 자를 예정)
        output_ids = model.generate(input_ids, max_new_tokens=512, temperature=0.7)
        
        # 입력 프롬프트 부분은 제외하고 답변만 추출
        response_ids = output_ids[0].tolist()[len(input_ids[0]):]
        response_text = enc.decode(response_ids)
        
    # [핵심] <|endoftext|> 토큰이 나오면 그 뒤는 싹둑 자르기
    if "<|endoftext|>" in response_text:
        response_text = response_text.split("<|endoftext|>")[0]
        
    return response_text.strip()

# ==========================================
# 4. 채팅 루프
# ==========================================
print("\n" + "="*40)
print("🤖 Alpaca-NanoGPT Chatbot is Ready!")
print("   (종료하려면 'quit' 입력)")
print("="*40 + "\n")

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Bye!")
            break
        
        if not user_input.strip():
            continue

        print("Thinking...", end="\r")
        response = generate_response(user_input)
        
        # 이전 출력 덮어쓰고 결과 출력
        print(f"Bot : {response}\n")
        print("-" * 40)
        
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        break
    except Exception as e:
        print(f"\nError: {e}")