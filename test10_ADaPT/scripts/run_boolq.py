import os
import re
import warnings
from datasets import load_dataset
from dotenv import load_dotenv
import sys

# 상위 폴더 src 참조
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
load_dotenv('../.env')

from src.controller import adapt
from src.controller_adv import adapt_adv
from src.tools import llm

if __name__ == "__main__":
    print("📥 BoolQ 데이터셋 로드 중...")
    dataset = load_dataset("google/boolq", split="validation")
    
    # yes/no 문제만 필터링
    ox_dataset = [item for item in dataset if item['answer'] in [True, False]]
    
    score, total = 0, 20  # 원하는 샘플 개수
    start_idx = 0
    
    for i, item in enumerate(ox_dataset[start_idx:start_idx+total], 1):
        question = item['question']
        context = item['passage']
        correct_answer = "yes" if item['answer'] else "no"
        
        print(f"\n{'='*80}\n📝 [문제 {i}/{total}] {question}\n🔑 정답: {correct_answer}\n{'='*80}")
        
        # ADaPT 에이전트 기반 실행
        success, info = adapt_adv(question, context, 0, 3)
        
        # LLM 최종 판단
        final_ans = llm.invoke(f"정보: {info}\n질문: {question}\n답: yes or no?").content.strip().lower()
        
        print(f"\n🤖 제출: {final_ans}")
        if correct_answer in final_ans:
            print("🎉 정답!"); score += 1
        else:
            print("❌ 오답")
    
    print(f"\n🏆 최종: {score}/{total}")