import os
import sys
os.environ['HF_HOME'] = '/scratch/x3326a36/hf_cache' # 적절히 바꾸기
# 현재 파일(scripts/...)의 부모 폴더(project_root)를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.model import GPT, GPTConfig
from transformers import GPT2LMHeadModel

def load_original_nanogpt():
    print(">>> 1. NanoGPT(=GPT-2 Small) 설정 준비 중...")
    # NanoGPT(GPT-2)의 공식 스펙입니다.
    config = GPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        bias=True,        # NanoGPT는 Bias를 사용합니다.
        vocab_size=50257, # [중요] OpenAI 공식 Vocab Size
        dropout=0.1
    )

    print(">>> 2. 님의 코드로 껍데기 모델 생성...")
    model = GPT(config)
    sd = model.state_dict()
    
    # 학습과 관련 없는 버퍼(.attn.bias 등) 제거
    sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')] 

    print(">>> 3. OpenAI에서 '진짜 NanoGPT(GPT-2)' 가중치 다운로드...")
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()

    # 불필요한 마스킹 버퍼 제거
    sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')] 
    
    # [핵심] OpenAI(Conv1D) -> NanoGPT(Linear) 가중치 모양 변환
    # 이 과정이 있어야 'NanoGPT'가 됩니다.
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    assert len(sd_keys_hf) == len(sd_keys), f"키 개수 불일치! HF: {len(sd_keys_hf)}, Mine: {len(sd_keys)}"

    print(">>> 4. 가중치 이식(Transplanting) 중...")
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # 전치(Transpose)가 필요한 레이어
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # 그대로 복사
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    print(">>> 5. 'last.pth'로 저장 (Trainer가 읽을 수 있게)")
    save_dir = "results/nanogpt_base/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "last.pth")

    # Trainer 포맷에 맞춤
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': {}, # 옵티마이저는 초기화
        'epoch': 0,
        'loss': -1,
        'config': config, 
    }
    
    torch.save(checkpoint, save_path)
    print(f">>> 성공! NanoGPT 베이스 모델이 저장되었습니다: {save_path}")
    print(">>> 이제 이 경로를 파인튜닝의 'out_dir'나 'init_from'으로 쓰면 됩니다!")

if __name__ == "__main__":
    load_original_nanogpt()