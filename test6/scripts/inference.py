import torch
import tiktoken
from src.model import GPT, GPTConfig # GPTConfig도 임포트 필요할 수 있음
import os

def generate_text(checkpoint_path, prompt, max_new_tokens=200, temperature=0.7):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f">>> Loading checkpoint from {checkpoint_path}")
    
    # [핵심 수정 1] weights_only=False 추가
    # 우리가 만든 파일이므로 안전합니다. 커스텀 클래스(GPTConfig)를 로드하기 위해 필수입니다.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # [핵심 수정 2] Config를 YAML이 아니라 체크포인트 내부에서 가져옴
    # load_nanogpt.py로 만든 파일은 'config' 키에 설정 객체가 들어있습니다.
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(">>> Config loaded directly from checkpoint.")
    else:
        # 만약 config가 없는 옛날 체크포인트라면 수동 설정 (NanoGPT 기본값)
        print(">>> Warning: Config not found in checkpoint. Using default GPT-2 config.")
        config = GPTConfig(
            block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768
        )

    # 2. 모델 초기화 및 가중치 로드
    model = GPT(config)
    
    # state_dict만 깔끔하게 로드
    state_dict = checkpoint.get('model_state', checkpoint)
    
    # 불필요한 접두사(_orig_mod 등) 처리
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 3. 토크나이저 설정
    enc = tiktoken.get_encoding("gpt2")
    
    if prompt == "":
        input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # 4. 텍스트 생성
    print(f">>> Generating based on prompt: '{prompt}'")
    print("-" * 50)
    
    with torch.no_grad():
        # top_k 등 추가적인 샘플링 옵션을 주면 더 자연스럽습니다
        y_gen = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50)
        generated_text = enc.decode(y_gen[0].tolist())
        print(generated_text)
    
    print("-" * 50)

if __name__ == "__main__":
    # [설정] 베이스 모델 경로 (load_nanogpt.py가 저장한 곳)
    # config.yaml은 필요 없습니다. last.pth 안에 다 있습니다.
    CHECKPOINT_PATH = "../results/gpt2_large_base/checkpoints/last.pth"
    user_prompt = "the captial of korea is"
    user_prompt = input()

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found.")
    else:
        generate_text(CHECKPOINT_PATH, user_prompt)