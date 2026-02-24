import os
# [중요] 모든 라이브러리 임포트 전 최상단에 경로 설정
os.environ['HF_HOME'] = '/scratch/x3326a36/hf_cache'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_raw_chat():
    # 1. 설정
    model_id = "Qwen/Qwen1.5-1.8B" # Chat 모델이 아닌 Base 모델 권장 (자동완성에 더 적합)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f">>> Loading model: {model_id}")
    
    # 2. 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("-" * 50)
    print("원하는 문장의 시작 부분을 입력하세요 (종료: 'exit')")
    print("-" * 50)

    while True:
        user_input = input("\n[Input]: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
            
        if not user_input.strip():
            continue

        # 3. 인코딩 (템플릿 없이 순수 텍스트만 처리)
        inputs = tokenizer(user_input, return_tensors="pt").to(device)

        # 4. 생성 (Raw Completion)
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

        # 5. 디코딩 (입력된 부분 제외하고 출력)
        # input_ids의 길이를 알아내어 그 이후부터 슬라이싱
        input_length = inputs.input_ids.shape[1]
        generated_tokens = output_ids[0][input_length:]
        
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"\n[Completion]: {response}")
        print("-" * 30)

if __name__ == "__main__":
    run_raw_chat()