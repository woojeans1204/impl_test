import wikipedia
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
import re

# =====================
# 1️⃣ LLM 초기화
# =====================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =====================
# 2️⃣ Tool 정의
# =====================
@tool
def calculate(expression: str) -> str:
    """수학 수식을 계산합니다. 예: calculate('3 * 5')"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def search_wikipedia(query: str) -> str:
    """위키백과에서 정보를 검색합니다."""
    wikipedia.set_lang("ko")
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Error: 여러 문서 후보: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Error: 검색 결과가 없습니다."
    except Exception as e:
        return f"Error: {e}"

tools = [calculate, search_wikipedia]

# =====================
# 3️⃣ ReAct 템플릿 프롬프트
# =====================
template = """당신은 질문을 해결하는 AI입니다. 다음 도구를 사용할 수 있습니다:

{tools}

규칙:
- Thought: [지금 생각하는 과정]
- Action: [사용할 도구 이름]
- Action Input: [도구 입력]
- Observation: [도구 결과]
- Final Answer: [최종 답변]

Action이 필요 없으면 바로 Final Answer 출력

Question: {input}
Thought:{agent_scratchpad}"""

prompt = ChatPromptTemplate.from_template(template)

# =====================
# 4️⃣ 에이전트 생성
# =====================
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,             # Thought/Action/Observation 단계 출력
    handle_parsing_errors=True
)

# =====================
# 5️⃣ Step별 ReAct 루프 시뮬레이션
# =====================
def react_simulation(question: str):
    print(f"[질문]\n{question}\n")
    messages = [{"role": "user", "content": question}]
    step = 0

    while step < 10:  # 최대 10번 루프
        step += 1
        # 에이전트 호출
        response = agent_executor.invoke({"input": question})
        assistant_msg = response["output"]
        
        print(f"\n=== Step {step} ===")
        print("[Thought / Action]")
        print(assistant_msg)
        
        # Final Answer 체크
        if "Final Answer:" in assistant_msg:
            print("\n[최종 답변]")
            break
        
        # Action 추출
        calc_match = re.search(r"Action:\s*calculate\((.*?)\)", assistant_msg)
        wiki_match = re.search(r"Action:\s*search_wikipedia\((.*?)\)", assistant_msg)
        
        observation = ""
        if calc_match:
            observation = calculate(calc_match.group(1))
        elif wiki_match:
            observation = search_wikipedia(wiki_match.group(1))
        else:
            observation = "Error: 올바른 Action을 찾지 못했습니다."
        
        print("[Observation]")
        print(observation)
        
        # 다음 루프를 위해 observation을 메시지로 추가
        messages.append({"role": "user", "content": f"Observation: {observation}"})

# =====================
# 6️⃣ 실행 예제
# =====================
if __name__ == "__main__":
    question = "현재 대한민국 대통령이 누구인지 위키백과에서 검색한 후, 그 사람의 나이를 0으로 나누면 어떤 결과가 나오는지 계산해보세요."
    react_simulation(question)