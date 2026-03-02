import re
import warnings
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")
load_dotenv('../.env')

# 1. LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 핫팟 QA 로드 및 yes/no 필터링
dataset = load_dataset("hotpot_qa", "distractor", split="validation")
ox_dataset = [item for item in dataset if item['answer'].lower() in ['yes', 'no']]

def run_baseline(dataset, total_test_count=10):
    score = 0
    for i, item in enumerate(ox_dataset[:total_test_count], 1):
        question = item['question']
        correct_answer = item['answer'].lower()
        
        print("\n" + "=" * 80)
        print(f"📝 [문제 {i}/{total_test_count}] {question}")
        print(f"🔑 [실제 정답] {correct_answer}")
        print("=" * 80)

        # LLM만 사용 → 단순 yes/no 답변
        prompt = f"""질문: {question}

주의: 질문에 대한 답을 부정확한 정보 없이 정확히 판단하세요.
오직 'yes' 또는 'no'로만 답하세요."""
        
        final_answer_msg = llm.invoke(prompt)
        final_answer = final_answer_msg.content.strip().lower()

        print(f"🤖 LLM 제출: {final_answer}")

        if re.fullmatch(rf"{correct_answer}", final_answer):
            print("🎉 [채점 결과] 정답입니다!")
            score += 1
        else:
            print("❌ [채점 결과] 오답입니다.")
        
        print(f"📊 현재 점수: {score} / {i}")

    print("\n" + "="*80)
    print(f"🏆 최종 점수: {score} / {total_test_count} ({score/total_test_count:.2%})")
    print("="*80)

if __name__ == "__main__":
    run_baseline(ox_dataset, total_test_count=10)