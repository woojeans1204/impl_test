import os
import re
import warnings
import concurrent.futures
from typing import List, Literal
from pydantic import BaseModel, Field
from datasets import load_dataset
from dotenv import load_dotenv

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

load_dotenv('../.env')

# 1. LLM 및 도구 초기화
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
    import wikipedia
    wikipedia.set_lang("en")
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Error: 동음이의어. 구체적으로 검색하세요. 후보: {e.options[:5]}"
    except Exception as e:
        return f"Error: 문서 없음. search_internet을 사용하세요. ({e})"

ddg_search = DuckDuckGoSearchRun()

@tool
def search_internet(query: str) -> str:
    """인터넷에서 최신 정보나 뉴스를 검색합니다. 위키백과에서 찾지 못했을 때 사용하세요."""
    def fetch():
        return ddg_search.invoke(query)
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fetch)
            return future.result(timeout=5)
    except concurrent.futures.TimeoutError:
        return "Error: 검색 서버 타임아웃."
    except Exception as e:
        return f"Error: {e}"

tools = [calculate, search_wikipedia, search_internet]
base_executor = create_react_agent(llm, tools)


# 2. Pydantic을 이용한 구조화된 출력 규격 정의
class EvaluatorOutput(BaseModel):
    is_success: bool = Field(description="태스크가 성공적으로 완수되었는지 여부 (True/False)")
    result_summary: str = Field(description="성공했다면 그 결과값, 실패했다면 실패 이유를 요약")

class PlannerOutput(BaseModel):
    operator: Literal["AND", "OR"] = Field(description="하위 태스크들의 논리적 관계. 순차적 실행이 필수면 AND, 여러 대안 중 하나만 성공해도 되면 OR.")
    sub_tasks: List[str] = Field(description="잘게 쪼개진 하위 태스크(sub-task) 목록. 최대 3개.")


# 3. ADaPT 핵심 3-모듈 정의 (Executor, Evaluator, Planner)

def run_executor(task: str, context: str) -> str:
    """Module 1. Executor: 오직 도구를 사용하여 정보만 탐색합니다. 스스로 평가하지 않습니다."""
    executor_system_prompt = """당신은 주어진 태스크를 수행하는 지식 탐색 실행기(Executor)입니다.
[가장 중요한 규칙]
절대 당신의 내장 지식을 사용하여 임의로 대답하지 마세요! 환각(Hallucination) 방지를 위해 반드시 제공된 검색 도구(search_wikipedia, search_internet)를 사용하여 사실을 직접 확인한 후에만 답변해야 합니다."""

    prompt = f"현재까지의 정보(Context):\n{context}\n\n위 정보를 바탕으로 다음 태스크를 도구를 사용해 완수하세요: {task}"
    
    try:
        result = base_executor.invoke({
            "messages": [
                SystemMessage(content=executor_system_prompt),
                HumanMessage(content=prompt)
            ]
        }, {"recursion_limit": 8})
        
        print("\n    [🔍 Executor 내부 사고 흐름]")
        for msg in result["messages"]:
            if msg.type in ["ai", "tool"]:
                msg.pretty_print()
        print("    " + "-"*40)
        
        return result["messages"][-1].content
    except Exception as e:
        return f"[실행기 에러] 단일 태스크로 처리하기에 너무 복잡하여 실패했습니다. (에러: {e})"

def run_evaluator(task: str, raw_output: str) -> EvaluatorOutput:
    """Module 2. Evaluator: Executor가 가져온 결과를 보고, 원래 태스크가 완수되었는지 객관적으로 평가합니다."""
    eval_prompt = f"원래 태스크: {task}\n\n실행기의 결과물: {raw_output}\n\n이 결과가 원래 태스크를 완벽하게 해결했나요? 아니면 에러가 나거나 정보가 부족한가요? 특히 단어의 의미에 주의하세요"
    evaluator = llm.with_structured_output(EvaluatorOutput)
    return evaluator.invoke(eval_prompt)

def run_planner(task: str, fail_reason: str) -> PlannerOutput:
    """Module 3. Planner: Evaluator가 실패 판정을 내렸을 때 호출되어 태스크를 분할합니다."""
    prompt = f"""원래 태스크: '{task}'
실패 원인: '{fail_reason}'

이 태스크를 더 작고 구체적인 하위 태스크 2~3개로 쪼개주세요. 순차적으로 모두 해야 한다면 AND, 대안적인 방법들이라면 OR 연산자를 선택하세요.
[가장 중요한 규칙]
태스크를 분할할 때는 '조사한다', '분석한다', '알아본다' 같은 추상적인 단어를 절대 사용하지 마세요!
반드시 실행기가 즉시 검색 도구에 입력할 수 있는 형태의 '구체적인 검색어 키워드' 또는 '명확한 행동' 단위로만 지시하세요.
(나쁜 예: "Fuding의 경제적 특성을 분석한다")
(좋은 예: "Fuding city tier level 검색", "Yingkou city population 검색")"""
    
    planner = llm.with_structured_output(PlannerOutput)
    return planner.invoke(prompt)

# 4. ✨ 오리지널 ADaPT 재귀 알고리즘 (Controller) ✨
def adapt(task: str, context: str, depth: int, max_depth: int = 3) -> tuple[bool, str]:
    indent = "  " * depth
    print(f"\n{indent}▶️ [Depth {depth}] ADaPT 호출: {task}")

    if depth > max_depth:
        print(f"{indent}❌ [Depth {depth}] 최대 깊이 초과로 중단.")
        return False, "최대 탐색 깊이 초과"

    # Step 1: Executor 실행
    print(f"{indent}🤖 [Depth {depth}] Executor 실행 중...")
    raw_output = run_executor(task, context)
    
    # Step 2: Evaluator 평가 (모듈 분리됨)
    print(f"{indent}⚖️ [Depth {depth}] Evaluator 평가 중...")
    eval_result = run_evaluator(task, raw_output)
    
    if eval_result.is_success:
        print(f"{indent}✅ [Depth {depth}] 태스크 성공: {eval_result.result_summary[:100]}...")
        return True, eval_result.result_summary

    # Step 3: 실패 시 Planner 호출
    print(f"{indent}⚠️ [Depth {depth}] 태스크 실패. Planner 호출 중... (이유: {eval_result.result_summary[:80]}...)")
    plan = run_planner(task, eval_result.result_summary)
    print(f"{indent}📋 [Depth {depth}] Planner 분할 완료 [{plan.operator}]: {plan.sub_tasks}")

    # Step 4: Controller 논리 전개
    if plan.operator == "AND":
        accumulated_context = context
        combined_result = ""
        for sub_task in plan.sub_tasks:
            sub_success, sub_result = adapt(sub_task, accumulated_context, depth + 1, max_depth)
            if not sub_success:
                print(f"{indent}❌ [Depth {depth}] AND 조건 실패: '{sub_task}'가 실패하여 중단합니다.")
                return False, f"'{sub_task}' 실패로 인한 중단"
            accumulated_context += f"\n[정보: {sub_result}]"
            combined_result += f" {sub_result}"
        return True, combined_result

    elif plan.operator == "OR":
        for sub_task in plan.sub_tasks:
            sub_success, sub_result = adapt(sub_task, context, depth + 1, max_depth)
            if sub_success:
                print(f"{indent}✅ [Depth {depth}] OR 조건 성공: '{sub_task}'가 성공하여 완료합니다.")
                return True, sub_result
        print(f"{indent}❌ [Depth {depth}] OR 조건 실패: 모든 대안이 실패했습니다.")
        return False, "모든 OR 대안 실패"

    return False, "알 수 없는 에러"


if __name__ == "__main__":
    print("📥 HotpotQA 데이터셋 로드 중... (오리지널 ADaPT 3-모듈 분리 버전)")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    ox_dataset = [item for item in dataset if item['answer'].lower() in ['yes', 'no']]
    
    score = 0
    total_test_count = 10
    
    for i, item in enumerate(ox_dataset[:total_test_count], 1):
        question = item['question']
        correct_answer = item['answer'].lower()
        
        print("\n" + "=" * 80)
        print(f"📝 [문제 {i}/{total_test_count}] {question}")
        print(f"🔑 [실제 정답] {correct_answer}")
        print("=" * 80)
        
        success, final_info = adapt(task=question, context="", depth=0, max_depth=3)
        
        print("\n[최종 판별 중...]")
        final_prompt = f"""질문: {question}

수집된 정보: {final_info}

[엄격한 팩트 체크 규칙]
1. 위 정보를 바탕으로 질문에 대한 답을 내리세요.
2. 만약 질문이 "A와 B는 모두 ~인가?"라면, A와 B가 '정확히' 그 정의에 부합하는지 깐깐하게 따지세요. 
3. (예를 들어, 질문이 Genus(속)를 묻는데 하나가 일반 명사(Common name)라면 답은 'no'입니다.)
4. 부가 설명 없이 오직 'yes' 또는 'no'로만 대답하세요."""
        final_answer_msg = llm.invoke(final_prompt)
        final_answer = final_answer_msg.content.strip().lower()
        
        print(f"🤖 최종 수집 정보 요약: {final_info[:200]}...")
        print(f"🤖 에이전트 최종 제출: {final_answer}")
        
        if re.search(r'\b' + re.escape(correct_answer) + r'\b', final_answer):
            print("🎉 [채점 결과] 정답입니다!")
            score += 1
        else:
            print("❌ [채점 결과] 오답입니다.")
        
        print(f"📊 현재 점수: {score} / {i}")