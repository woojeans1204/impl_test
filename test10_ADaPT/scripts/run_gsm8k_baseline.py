import os
import re
import warnings
from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd

warnings.filterwarnings("ignore")
load_dotenv('../.env')

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def extract_ground_truth(text: str) -> str:
    match = re.search(r'####\s*(-?\d+)', text)
    if match: return match.group(1)
    return text.split()[-1]

def extract_predicted_number(text: str) -> str:
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    if numbers: return numbers[-1]
    return text

def run_baseline(dataset, total_test_count=10):
    score = 0
    start_idx = 20
    results = []
    
    for i, item in enumerate(dataset.select(range(start_idx, start_idx+total_test_count)), 1):
        question = item['question']
        correct_answer = extract_ground_truth(item['answer'])
        
        print("\n" + "="*80)
        print(f"📝 [문제 {i}/{total_test_count}] {question}")
        print(f"🔑 [실제 정답] {correct_answer}")
        print("="*80)
        
        prompt = f"""질문: {question}

주의: 문제를 단계별로 풀이하고, 가장 마지막에는 '정답: [숫자]' 형식으로 답만 명확히 적어주세요."""
        
        final_answer_msg = llm.invoke(prompt)
        final_answer_raw = final_answer_msg.content.strip()
        final_answer = extract_predicted_number(final_answer_raw)
        
        print(f"🤖 LLM 제출 과정:\n{final_answer_raw}\n")
        print(f"🎯 추출된 숫자: {final_answer}")
        
        is_correct = (correct_answer == final_answer)
        if is_correct:
            print("🎉 정답!")
            score += 1
        else:
            print("❌ 오답")
        
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "llm_raw": final_answer_raw,
            "llm_extracted": final_answer,
            "is_correct": is_correct
        })
        
        print(f"📊 현재 점수: {score} / {i}")
    
    print("\n" + "="*80)
    print(f"🏆 최종 점수: {score} / {total_test_count} ({score/total_test_count:.2%})")
    print("="*80)
    
    return results

if __name__ == "__main__":
    print("📥 GSM8K 데이터셋 로드 중...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    total_test_count = 10
    results = run_baseline(dataset, total_test_count=total_test_count)
    
    df = pd.DataFrame(results)
    df.to_csv("gsm8k_llm_baseline_results.csv", index=False, encoding="utf-8-sig")
    print("✅ 결과가 'gsm8k_llm_baseline_results.csv'로 저장되었습니다.")