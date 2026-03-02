import os
import re
import warnings
from datasets import load_dataset
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv('../.env')

from langchain_openai import ChatOpenAI

# 1️⃣ LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2️⃣ BoolQ 데이터셋 로드
print("📥 BoolQ 데이터셋 로드 중...")
dataset = load_dataset("google/boolq", split="validation")

# yes/no 문제만 필터링
ox_dataset = [item for item in dataset if item['answer'] in [True, False]]

def run_baseline(dataset, total_test_count=10):
    score = 0
    results = []
    
    for i, item in enumerate(ox_dataset[:total_test_count], 1):
        question = item['question']
        passage = item['passage']
        correct_answer = "yes" if item['answer'] else "no"
        
        print("\n" + "="*80)
        print(f"📝 [문제 {i}/{total_test_count}] {question}")
        print(f"🔑 [실제 정답] {correct_answer}")
        print("="*80)
        
        # 3️⃣ LLM 단독 yes/no 판단
        prompt = f"""질문: {question}
        
본문: {passage}

주의: 위 내용을 바탕으로 정확히 판단하고, 답은 'yes' 또는 'no'만 출력하세요."""
        
        final_answer_msg = llm.invoke(prompt)
        final_answer = final_answer_msg.content.strip().lower()
        
        print(f"🤖 LLM 제출: {final_answer}")
        
        is_correct = correct_answer == final_answer
        if is_correct:
            print("🎉 정답!")
            score += 1
        else:
            print("❌ 오답")
        
        results.append({
            "question": question,
            "passage": passage,
            "correct_answer": correct_answer,
            "llm_answer": final_answer,
            "is_correct": is_correct
        })
        
        print(f"📊 현재 점수: {score} / {i}")
    
    print("\n" + "="*80)
    print(f"🏆 최종 점수: {score} / {total_test_count} ({score/total_test_count:.2%})")
    print("="*80)
    
    return results

if __name__ == "__main__":
    total_test_count = 20  # 원하는 샘플 개수
    results = run_baseline(ox_dataset, total_test_count=total_test_count)
    