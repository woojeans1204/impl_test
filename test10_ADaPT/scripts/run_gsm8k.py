import os
import re
import warnings
from datasets import load_dataset
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
load_dotenv('../.env')

from src.controller_adv import adapt_adv
from src.tools import llm

def extract_ground_truth(text: str) -> str:
    match = re.search(r'####\s*(-?\d+)', text)
    if match: return match.group(1)
    return text.split()[-1]

def extract_predicted_number(text: str) -> str:
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    if numbers: return numbers[-1]
    return text

if __name__ == "__main__":
    print("📥 GSM8K 데이터셋 로드 중...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    score, total = 0, 10
    start_idx = 0
    
    for i, item in enumerate(dataset.select(range(start_idx, start_idx+total)), 1):
        question = item['question']
        correct_answer = extract_ground_truth(item['answer'])
        
        print(f"\n{'='*80}\n📝 [문제 {i}/{total}] {question}\n🔑 정답: {correct_answer}\n{'='*80}")
        
        success, final_summary = adapt_adv(question, "", 0, 3)

        if success:
            # 지저분한 Trace 로그는 무시하고, 'final_summary'(판사의 최종 판결문)만 사용
            extract_prompt = f"다음 질문 {question}에 대한 자연어로 된 답에서 숫자를 찾아서 숫자만 파싱해 정답을 직접 맞춰볼 수 있게 콤마 다 떼고 아라비아 숫자: {final_summary}"
            print(final_summary)
            final_ans = llm.invoke(extract_prompt).content
            
            # 숫자만 필터링 (예: "$70,000" -> "70000")
            # final_ans = "".join(filter(lambda x: x.isdigit() or x == '.', final_ans))
            print(f"제출 숫자: {final_ans}")
        
        if correct_answer == final_ans:
            print("🎉 정답!")
            score += 1
        else:
            print("❌ 오답")
    
    print(f"\n🏆 최종: {score}/{total} ({score/total:.2%})")