import os
import sys
import json
import requests
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# 상위 폴더의 src를 참조하기 위해 path 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools import llm  # 이미 ChatOpenAI 초기화된 llm

load_dotenv('../.env')

def load_data():
    print("📥 Ko-StrategyQA 로드 중...")
    url = "https://huggingface.co/datasets/NomaDamas/Ko-StrategyQA/resolve/main/ko-strategy-qa_train.json"
    data = requests.get(url).json()
    
    records = []
    for doc_id, doc_info in data.items():
        doc_info['id'] = doc_id
        for key, value in doc_info.items():
            if isinstance(value, (list, dict)):
                doc_info[key] = json.dumps(value, ensure_ascii=False)
        records.append(doc_info)
        
    return Dataset.from_pandas(pd.DataFrame(records))

def run_baseline(dataset, start_idx=0, total=5):
    score = 0
    for i, item in enumerate(dataset.select(range(start_idx, start_idx+total)), 1):
        question = item['question']
        correct_answer = "yes" if item['answer'] else "no"
        
        print(f"\n{'='*80}\n📝 [문제 {i}/{total}] {question}\n🔑 정답: {correct_answer}\n{'='*80}")
        
        # adapt 없이 바로 LLM 호출
        final_ans = llm.invoke(f"질문: {question}\n답: yes or no?").content.strip().lower()
        
        print(f"\n🤖 모델 답변: {final_ans}")
        if correct_answer in final_ans:
            print("🎉 정답!")
            score += 1
        else:
            print("❌ 오답")
    
    print(f"\n🏆 최종 정확도: {score}/{total} = {score/total:.2%}")

if __name__ == "__main__":
    dataset = load_data()
    run_baseline(dataset, start_idx=0, total=20)