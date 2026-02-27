import os
os.environ['HF_HOME'] = '/scratch/x3326a36/hf_cache'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 캐시 경로 설정 (서버 환경에 맞게 수정하세요)

def generate_text(prompt, model_id = "Qwen/Qwen1.5-1.8B"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">>> 1. Loading tokenizer and model from Hugging Face: {model_id}")
    
    # [핵심 1] 토크나이저와 모델을 한 번에 로드합니다. (Load 스크립트 불필요!)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # V100 GPU 등에 맞게 메모리를 아끼려면 torch_dtype=torch.float16을 줍니다.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16, 
        device_map="auto" # GPU가 여러 개면 알아서 분배해 줍니다.
    )
    
    # [핵심 2] Qwen은 특유의 ChatML 포맷을 써야 말을 잘 듣습니다.
    # apply_chat_template이 이 복잡한 포맷팅을 자동으로 해줍니다.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"\n>>> 2. Formatted Prompt:\n{text}")
    print("-" * 50)
    
    # 텐서로 변환 후 GPU로 이동
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 3. 텍스트 생성
    print(">>> 3. Generating response...\n")
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id # 경고 메시지 방지용
        )

    # [핵심 3] 입력 프롬프트 길이를 계산해서, 모델이 '새로 생성한' 답변만 잘라냅니다.
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    user_prompt = input("질문을 입력하세요 (예: 한국의 수도는?): ")
    if user_prompt.strip():
        generate_text(user_prompt)