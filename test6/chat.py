import torch
import tiktoken
from src.model import GPT

# 1. 설정 (학습 끝난 모델 경로)
CHECKPOINT_PATH = "results/nanogpt_alpaca_finetune/checkpoints/last.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. 모델 로드 (옵션 주의!)
print(f"Loading model from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

# 3. 토크나이저
enc = tiktoken.get_encoding("gpt2")

def generate_response(user_input):
    # [핵심] Alpaca 프롬프트 포맷 씌우기
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_input}

### Response:
"""
    # 인코딩 & GPU 전송
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # 생성 (답변 부분만)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.7)
        # 프롬프트 길이를 제외하고 답변만 잘라냄
        response_ids = output_ids[0].tolist()[len(input_ids[0]):]
        response_text = enc.decode(response_ids)
        
    return response_text.strip()

# 4. 채팅 루프
print("\n" + "="*30)
print("🤖 Alpaca-NanoGPT Chatbot")
print("quit를 입력하면 종료합니다.")
print("="*30 + "\n")

while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break
    
    response = generate_response(user_input)
    print(f"Bot : {response}\n")