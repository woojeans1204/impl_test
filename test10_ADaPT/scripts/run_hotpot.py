import os
import re
import warnings
from datasets import load_dataset
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
load_dotenv('../.env')

from src.controller import adapt
from src.controller_adv import adapt_adv
from src.tools import llm

if __name__ == "__main__":
    print("📥 HotpotQA 데이터셋 로드 중...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # yes/no 문제만 필터링
    ox_dataset = [item for item in dataset if item['answer'].lower() in ['yes', 'no']]
    
    score, total = 0, 10  # 원하는 샘플 개수
    start_idx = 0
    
    for i, item in enumerate(ox_dataset[start_idx:start_idx+total], 1):
        question = item['question']
        correct_answer = item['answer'].lower()
        
        print(f"\n{'='*80}\n📝 [문제 {i}/{total}] {question}\n🔑 정답: {correct_answer}\n{'='*80}")
        
        # ADaPT 에이전트 기반 실행
        success, info = adapt_adv(question, "", 0, 3)
        
        # LLM 최종 판단
        final_ans = llm.invoke(f"정보: {info}\n질문: {question}\n답: yes or no?").content.strip().lower()
        
        print(f"\n🤖 제출: {final_ans}")
        if correct_answer in final_ans:
            print("🎉 정답!"); score += 1
        else:
            print("❌ 오답")
    
    print(f"\n🏆 최종: {score}/{total}")