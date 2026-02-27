import os
import wikipedia
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
    wikipedia.set_lang("ko")
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Error: {e}"

ddg_search = DuckDuckGoSearchRun()

@tool
def search_internet(query: str) -> str:
    """인터넷에서 최신 정보나 뉴스를 검색합니다. 위키백과에 없는 최신 사실을 찾을 때 사용하세요."""
    try:
        return ddg_search.invoke(query)
    except Exception as e:
        return f"Error: 인터넷 검색 중 오류 발생 ({e})"

tools = [calculate, search_wikipedia, search_internet]

agent = create_react_agent(llm, tools)

system_prompt = """당신은 질문에 답하기 위해 도구를 사용하는 AI 비서입니다.
매우 중요한 규칙: 항상 모든 경우에 도구를 호출하기 전에, 반드시 왜 이 도구를 호출하는지 'Thought:' 뒤에 당신의 '사고 과정(Thought)'을 텍스트로 먼저 설명하세요.
예시: 'Thought: 최신 날씨를 알기 위해 인터넷 검색을 해야겠습니다.'"""

if __name__ == "__main__":
    question = "현재 대한민국 부산의 날씨를 인터넷으로 검색한 후, 그 섭씨 온도를 화씨 온도로 변환하는 수식을 계산해보세요."
    question = "현재 대한민국 대통령이 누구인지 위키백과에서 검색한 후, 그 사람의 나이를 0으로 나누면 어떤 결과가 나오는지 계산해보세요."

    print(f"--- 질문: {question} ---\n")
    
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
    }
    
    for step in agent.stream(inputs):
        for node_name, node_state in step.items():
            print(f"\n[{node_name.upper()} 단계 실행]")
            for msg in node_state["messages"]:
                msg.pretty_print()