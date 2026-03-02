import os
import re
import sys
import warnings
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
load_dotenv('../.env')

from src.controller_adv import adapt_adv
from src.tools import llm

def extract_ground_truth_math(text: str) -> str:
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

if __name__ == "__main__":
    print("📥 MATH 데이터셋 로드 중...")
    dataset = load_dataset("EleutherAI/hendrycks_math", 'counting_and_probability')['test']
    
    score, total = 0, 10
    start_idx = 5
    
    for i, item in enumerate(dataset.select(range(start_idx, start_idx + total)), 1):
        question = item['problem']
        solution = item['solution']
        correct_answer = extract_ground_truth_math(solution)
        
        print(f"\n{'='*80}\n📝 [문제 {i}/{total}] {question}\n🔑 정답: {correct_answer}\n{'='*80}")
        
        # ADaPT 실행
        success, final_summary = adapt_adv(question, "", 0, 3)
        final_ans = "Fail" 

        if success:
            print(f"\n[판사 요약문]: {final_summary}")
            # MATH는 문자열/수식 정답이 섞여 있으므로 파싱 전략 수정
            extract_prompt = f"다음 판결문에서 최종 정답에 해당하는 수치나 수식만 추출해. 불필요한 문장 없이 정답만 '\\boxed{{정답}}' 형식으로 출력해.\n판결문: {final_summary}"
            
            final_ans_raw = llm.invoke(extract_prompt).content
            final_ans = extract_ground_truth_math(final_ans_raw) 
            
            print(f"🎯 제출 정답: {final_ans}")
        else:
            print(f"⚠️ 에이전트 풀이 실패: {final_summary}")
        
        # LaTeX 포맷 차이로 인한 오답 처리를 최소화하기 위해 공백 제거 후 비교
        is_correct = check_answer_equivalence(correct_answer, final_ans)
        if is_correct:
            print("🎉 정답!")
            score += 1
        else:
            print("❌ 오답")
        
        print(f"📊 현재 점수: {score} / {i}")
    
    print(f"\n{'='*80}\n🏆 최종 점수: {score}/{total} ({score/total:.2%})\n{'='*80}")