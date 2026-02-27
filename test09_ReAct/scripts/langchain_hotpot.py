import os
import wikipedia
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os

load_dotenv('../.env')
# print(os.environ.get("OPENAI_API_KEY"))  # 확인용
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 1. LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 도구 정의
@tool
def calculate(expression: str) -> str:
    """수학 수식을 계산합니다. 예: calculate('3 * 5')"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def search_wikipedia(query: str) -> str:
    """위키백과에서 정보를 검색합니다. 공식적인 개념이나 인물 정보를 찾을 때 사용하세요."""
    # HotpotQA는 영어 질문이므로 영어 위키백과로 검색해야 정확도가 높습니다.
    wikipedia.set_lang("en")
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Error: 동음이의어 문서가 많습니다. 더 구체적인 검색어를 사용하세요. 후보: {e.options[:5]}"
    except Exception as e:
        return f"Error: 문서가 없습니다. search_internet 도구를 사용해보세요. ({e})"

ddg_search = DuckDuckGoSearchRun()

@tool
def search_internet(query: str) -> str:
    """인터넷에서 최신 정보나 뉴스를 검색합니다. 위키백과에서 찾지 못했을 때 사용하세요."""
    try:
        return ddg_search.invoke(query)
    except Exception as e:
        return f"Error: 인터넷 검색 중 오류 발생 ({e})"

tools = [calculate, search_wikipedia, search_internet]

# 3. 에이전트 생성
agent = create_react_agent(llm, tools)

# 4. 프롬프트 설정 (사고 과정 강제 및 무한 루프 방지)
system_prompt = """당신은 질문에 답하기 위해 도구를 사용하는 AI 비서입니다.
매우 중요한 규칙:
0. 지식은 반드시 도구를 이용해 검색하세요. 환각 효과를 막기 위해 필수적인 과정입니다.
1. 도구를 호출하기 전, 반드시 텍스트 본문(content)에 'Thought: [여기에 한글로 사고 과정 작성]'을 적으세요. 절대 생략하면 안 됩니다. 반드시 한글로 작성하세요.
2. 도구 실행 결과가 Error로 나오면, 절대 동일한 검색어로 똑같은 도구를 반복해서 호출하지 마세요. 검색어를 바꾸거나 search_internet 도구로 전환하세요.
3. 이 문제는 OX(Yes/No) 벤치마크 테스트입니다. 조사가 끝나면 최종 답변으로 부가 설명 없이 오직 'yes' 또는 'no' 중 하나만 소문자로 출력하세요.
"""

if __name__ == "__main__":
    print("📥 HotpotQA 데이터셋 로드 중... (최초 1회 다운로드 필요)")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    ox_dataset = [item for item in dataset if item['answer'].lower() in ['yes', 'no']]
    print(f"✅ 전체 데이터 중 OX(yes/no) 문제 {len(ox_dataset)}개를 필터링했습니다.\n")
    
    score = 0  # 맞춘 문제 수를 누적할 변수
    total_test_count = 5 # 테스트할 문제 수
    
    for i, item in enumerate(ox_dataset[10:10+total_test_count], 1):
        question = item['question']
        correct_answer = item['answer'].lower()
        
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
        
        final_answer = ""
        
        # recursion_limit을 15로 설정하여 에이전트가 무한 루프에 빠지는 것을 강제로 막습니다.
        try:
            for step in agent.stream(inputs, {"recursion_limit": 15}):
                for node_name, node_state in step.items():
                    print(f"\n[{node_name.upper()} 단계 실행]")
                    
                    last_msg = node_state["messages"][-1]
                    last_msg.pretty_print()
                    
                    # 에이전트 단계이고 도구 호출이 없다면 최종 답변으로 간주합니다.
                    if node_name == "agent" and not last_msg.tool_calls:
                        final_answer = last_msg.content.strip().lower()
        except Exception as e:
            print(f"\n⚠️ 실행 중단 (무한 루프 또는 에러): {e}")
            final_answer = "error"

        # 자동 채점 로직
        print("\n" + "-" * 70)
        print(f"🤖 에이전트 최종 제출: {final_answer}")
        
        # 정답에 yes/no가 포함되어 있는지 확인하여 채점합니다.
        if correct_answer in final_answer:
            print("🎉 [채점 결과] 정답입니다!")
            score += 1
        else:
            print("❌ [채점 결과] 오답입니다.")
        
        print(f"📊 현재 점수: {score} / {i}")
        print("-" * 70)
        
        # if i < total_test_count:
        #     input("\n⌨️ 다음 문제로 넘어가려면 Enter 키를 누르세요... (종료하려면 Ctrl+C)")

    print("\n" + "=" * 70)
    print(f"🏆 최종 테스트 완료! 정답률: {score}/{total_test_count} ({score/total_test_count*100:.1f}%)")
    print("=" * 70)