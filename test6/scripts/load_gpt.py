import os
os.environ['HF_HOME'] = '/scratch/x3326a36/hf_cache' # 적절히 바꾸기
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import torch
from src.model import GPT, GPTConfig
from transformers import GPT2LMHeadModel

def load_any_gpt2(model_type="gpt2-medium"):
    """
    model_type: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl' 중 선택
    """
    print(f">>> 1. {model_type} 설정 준비 중...")
    
    # 모델 타입에 따른 스펙 설정
    configs = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 355M
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1.5B
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}")

    config_args = configs[model_type]
    config_args.update(dict(vocab_size=50257, block_size=1024, bias=True, dropout=0.1))
    
    config = GPTConfig(**config_args)

    print(f">>> 2. {model_type} 껍데기 모델 생성...")
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')] 

    print(f">>> 3. HuggingFace에서 '{model_type}' 원본 가중치 다운로드...")
    # 모델 크기에 따라 수 GB의 용량이 필요합니다.
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')] 
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    assert len(sd_keys_hf) == len(sd_keys), f"키 개수 불일치! HF: {len(sd_keys_hf)}, Mine: {len(sd_keys)}"

    print(f">>> 4. 가중치 이식(Transplanting) 중...")
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    # 저장 경로 설정
    save_dir = f"results/{model_type.replace('-', '_')}_base/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "last.pth")

    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': {},
        'epoch': 0,
        'loss': -1,
        'config': config, 
    }
    
    torch.save(checkpoint, save_path)
    print(f">>> 성공! {model_type} 베이스 모델 저장됨: {save_path}")

if __name__ == "__main__":
    # 원하는 모델로 바꿔서 실행하세요!
    # load_any_gpt2("gpt2-large")
    load_any_gpt2("gpt2-xl")