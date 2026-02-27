import os
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
import re
from dotenv import load_dotenv

load_dotenv('../.env')

# 1. LLM 초기화 (수학 문제이므로 온도를 0으로 하여 일관성 유지)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 도구 정의 (수학 벤치마크이므로 계산기 도구만 사용합니다)
@tool
def calculate(expression: str) -> str:
    """수학 수식을 계산합니다. 예: calculate('3 * 5 + (10 / 2)')"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [calculate]

# 3. 에이전트 생성
agent = create_react_agent(llm, tools)

# 4. 프롬프트 설정 (수학 문제 풀이에 맞게 조정)
system_prompt = """당신은 수학 문제를 푸는 AI 비서입니다.
반드시 다음 규칙을 엄격하게 지키세요:
1. 계산이 필요하면 절대 암산하지 말고, 제공된 `calculate` 도구를 사용하세요.
2. 텍스트로 ```json ... ``` 같은 코드 블록을 직접 작성해서 도구를 호출하려 하지 마세요. 도구는 반드시 내장된 네이티브 함수 호출(Function Calling) 기능으로만 실행해야 합니다.
3. 도구를 호출하기 전, 어떤 계산을 할 것인지 한글로 풀이 과정을 먼저 설명하세요.
4. 모든 계산이 끝나면 맨 마지막 줄에 오직 '최종 정답: [숫자]' 형식으로만 정답을 출력하세요.
"""

if __name__ == "__main__":
    print("📥 GSM8K 데이터셋 로드 중... (최초 1회 다운로드 필요)")
    # GSM8K 데이터셋의 메인(main) 구성을 가져옵니다.
    dataset = load_dataset("gsm8k", "main", split="test")
    
    score = 0
    total_test_count = 10 # 테스트할 문제 수
    
    for i, item in enumerate(dataset.select(range(total_test_count)), 1):
        question = item['question']
        # GSM8K의 정답은 항상 '#### [숫자]' 형식으로 끝납니다. 이 숫자만 추출합니다.
        correct_answer = item['answer'].split("####")[1].strip()
        
        print("\n" + "=" * 70)
        print(f"📝 [문제 {i}/{total_test_count}] {question}")
        print(f"🔑 [실제 정답] {correct_answer}")
        print("=" * 70)
        
        inputs = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
        }
        
        final_answer_text = ""
        
        try:
            for step in agent.stream(inputs, {"recursion_limit": 15}):
                for node_name, node_state in step.items():
                    print(f"\n[{node_name.upper()} 단계 실행]")
                    
                    last_msg = node_state["messages"][-1]
                    last_msg.pretty_print()
                    
                    if node_name == "agent" and not last_msg.tool_calls:
                        final_answer_text = last_msg.content.strip()
        except Exception as e:
            print(f"\n⚠️ 실행 중단: {e}")
            final_answer_text = "error"

        # 에이전트의 답변에서 '최종 정답: [숫자]' 부분을 정규표현식으로 추출해 채점합니다.
        extracted_answer = None
        match = re.search(r"최종\s*정답:\s*([0-9,.-]+)", final_answer_text)
        if match:
            # 쉼표(,) 같은 천단위 구분기호 제거 후 비교
            extracted_answer = match.group(1).replace(",", "")
            
        print("\n" + "-" * 70)
        print(f"🤖 에이전트가 도출한 숫자: {extracted_answer}")
        
        if extracted_answer == correct_answer:
            print("🎉 [채점 결과] 정답입니다!")
            score += 1
        else:
            print("❌ [채점 결과] 오답입니다.")
        
        print(f"📊 현재 점수: {score} / {i}")
        print("-" * 70)
        
        if i < total_test_count:
            input("\n⌨️ 다음 문제로 넘어가려면 Enter 키를 누르세요... (종료하려면 Ctrl+C)")

    print("\n" + "=" * 70)
    print(f"🏆 GSM8K 테스트 완료! 정답률: {score}/{total_test_count} ({score/total_test_count*100:.1f}%)")
    print("=" * 70)