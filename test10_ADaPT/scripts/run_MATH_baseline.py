import os
import re
import warnings
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")
load_dotenv('../.env')

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def extract_ground_truth_math(text: str) -> str:
    # MATH 데이터셋은 정답이 \boxed{정답} 형태임 (1단계 중첩 괄호까지 허용)
    matches = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', text)
    if matches: 
        return matches[-1].strip()
    return text.strip().split()[-1]
def check_answer_equivalence(correct: str, predicted: str) -> bool:
    if predicted == "Fail" or not predicted:
        return False
    prompt = f"두 수학 정답이 수학적으로 완전히 동일한지(동치인지) 판별하세요.\n실제 정답: {correct}\n제출된 정답: {predicted}\n형식만 다르고 같은 값이면 'True', 다르면 'False'만 출력하세요."
    response = llm.invoke(prompt).content.strip().lower()
    return 'true' in response

def run_baseline(dataset, start_idx=0, total_test_count=10):
    score = 0
    
    for i, item in enumerate(dataset.select(range(start_idx, start_idx + total_test_count)), 1):
        question = item['problem'] # MATH 데이터셋은 'question' 대신 'problem'
        solution = item['solution']
        correct_answer = extract_ground_truth_math(solution)
        
        print("\n" + "="*80)
        print(f"📝 [문제 {i}/{total_test_count}] {question}")
        print(f"🔑 [실제 정답] {correct_answer}")
        print("="*80)
        
        prompt = f"""질문: {question}

주의: 문제를 단계별로 풀이하고, 가장 마지막에는 반드시 '\\boxed{{정답}}' 형식으로 답만 명확히 적어주세요."""
        
        final_answer_msg = llm.invoke(prompt)
        final_answer_raw = final_answer_msg.content.strip()
        final_answer = extract_ground_truth_math(final_answer_raw)
        
        print(f"🤖 LLM 제출 과정:\n{final_answer_raw}\n")
        print(f"🎯 제출 숫자/수식: {final_answer}")
        
        # MATH는 수식이 많아 정확한 문자열 일치가 어려울 수 있으나, 일차적으로 완전 일치 검사
        is_correct = check_answer_equivalence(correct_answer, final_answer)
        if is_correct:
            print("🎉 정답!")
            score += 1
        else:
            print("❌ 오답")
        
        print(f"📊 현재 점수: {score} / {i}")
    
    print("\n" + "="*80)
    print(f"🏆 최종 점수: {score} / {total_test_count} ({score/total_test_count:.2%})")
    print("="*80)

if __name__ == "__main__":
    print("📥 MATH 데이터셋 로드 중...")
    dataset = load_dataset("EleutherAI/hendrycks_math", 'algebra')['test']
    
    run_baseline(dataset, start_idx=0, total_test_count=10)