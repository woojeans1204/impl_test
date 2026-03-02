import os
import re
import warnings
import concurrent.futures
from typing import List, Literal
from pydantic import BaseModel, Field
from datasets import load_dataset
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_experimental.utilities.python import PythonREPL
load_dotenv('../.env')

# 1. LLM 및 고도화된 도구 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# PythonREPL: 에이전트가 직접 코드를 짜서 논리/계산을 검증하게 함
python_repl = PythonREPL()

@tool
def run_python_code(code: str) -> str:
    """파이썬 코드를 실행하여 복잡한 계산이나 논리 검증을 수행합니다. 예: 수식 계산, 날짜 비교 등"""
    try:
        # 보안을 위해 출력 제한 및 결과 반환
        return python_repl.run(code)
    except Exception as e:
        return f"Python Error: {e}"

@tool
def search_wikipedia(query: str) -> str:
    """위키백과에서 정보를 검색합니다. 공식적인 데이터나 인물/사건 정보를 찾을 때 필수입니다."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    try:
        return wiki.run(query)
    except Exception as e:
        return f"Wiki Error: {e}"

ddg_search = DuckDuckGoSearchRun()

@tool
def search_internet(query: str) -> str:
    """인터넷에서 최신 정보나 뉴스, 상식을 검색합니다. 위키백과에 없는 구체적 사실 확인 시 사용하세요."""
    def fetch():
        return ddg_search.invoke(query)
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fetch)
            return future.result(timeout=10)
    except Exception as e:
        return f"Internet Search Error: {e}"

tools = [run_python_code, search_wikipedia, search_internet]
# Executor 내부 에이전트의 한계를 3회로 줄여, 억지로 Planner를 깨우도록 설정(Decomposition 유도)
base_executor = create_react_agent(llm, tools)


# 2. Pydantic 규격
class EvaluatorOutput(BaseModel):
    is_success: bool = Field(description="태스크가 성공적으로 완수되었는지 여부")
    result_summary: str = Field(description="성공/실패 이유 요약")

class PlannerOutput(BaseModel):
    operator: Literal["AND", "OR"] = Field(description="논리 관계")
    sub_tasks: List[str] = Field(description="구체적인 실행 단위 키워드 중심 하위 작업 (최대 3개)")


# 3. 3-모듈 정의
def run_executor(task: str, context: str) -> str:
    """Module 1: Executor - 검색 및 코드 실행 수행"""
    system_prompt = """당신은 지식 탐색 실행기입니다.
[규칙] 내장 지식 사용 금지. 복잡한 비교나 계산은 run_python_code를 사용해 코드로 검증하세요."""
    prompt = f"컨텍스트: {context}\n임무: {task}"
    try:
        result = base_executor.invoke({
            "messages": [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        }, {"recursion_limit": 10}) 
        
        print("\n    [🔍 Executor 사고 흐름]")
        for msg in result["messages"]:
            if msg.type in ["ai", "tool"]: msg.pretty_print()
        return result["messages"][-1].content
    except Exception as e:
        return f"Executor Failure: {e}"

def run_evaluator(task: str, raw_output: str) -> EvaluatorOutput:
    """Module 2: Evaluator - 결과 검증"""
    eval_prompt = f"태스크: {task}\n결과물: {raw_output}\n\n위 결과가 태스크를 완벽한 팩트에 기반해 해결했나요? 엄격하게 논리적으로 추론할 수 있는지 상식적인 선에서 질문의 의도를 파악해 객관적으로 체크하세요."
    evaluator = llm.with_structured_output(EvaluatorOutput)
    # 1. LLM 호출 (결과는 EvaluatorOutput 객체)
    result = evaluator.invoke(eval_prompt)
    
    # 2. 결과가 실패인 경우 이유 출력 (객체 속성으로 접근)
    if not result.is_success:
        print(f"\n    [⚠️ Evaluator 판단: 실패]")
        print(f"    사유: {result.result_summary}")
        
    return result

def run_planner(task: str, fail_reason: str) -> PlannerOutput:
    """Module 3: Planner - 전략적 분할"""
    prompt = f"""태스크: '{task}' / 실패이유: '{fail_reason}'
이 전략적 질문을 해결하기 위한 '팩트 체크 단계'로 쪼개세요. 3단계 이하로 쪼개세요.
추상적 단어 금지. 예: '아리스토텔레스 생존 시기 검색', '노트북 발명 년도 검색'"""
    planner = llm.with_structured_output(PlannerOutput)
    return planner.invoke(prompt)


# 4. ADaPT 컨트롤러 (재귀)
def adapt(task: str, context: str, depth: int, max_depth: int = 3) -> tuple[bool, str]:
    indent = "  " * depth
    print(f"\n{indent}▶️ [Depth {depth}] ADaPT: {task}")
    if depth > max_depth: return False, "Max depth reached"

    raw_output = run_executor(task, context)
    eval_result = run_evaluator(task, raw_output)
    
    if eval_result.is_success:
        print(f"{indent}✅ 성공: {eval_result.result_summary[:50]}...")
        return True, eval_result.result_summary

    print(f"{indent}⚠️ 실패. Planner 분할 중...")
    plan = run_planner(task, eval_result.result_summary)
    print(f"{indent}📋 분할 [{plan.operator}]: {plan.sub_tasks}")

    if plan.operator == "AND":
        acc_context, combined = context, ""
        for sub in plan.sub_tasks:
            s, r = adapt(sub, acc_context, depth + 1, max_depth)
            if not s: return False, f"Fail at {sub}"
            acc_context += f"\n[{r}]"
            combined += f" {r}"
        return True, combined
    elif plan.operator == "OR":
        for sub in plan.sub_tasks:
            s, r = adapt(sub, context, depth + 1, max_depth)
            if s: return True, r
        return False, "All OR failed"
    return False, "Error"

def load_data():
    import requests
    import json
    from datasets import Dataset  
    import pandas as pd

    print("📥 Ko-StrategyQA 로드 중 (JSON 파싱 및 구조 평탄화)...")
    url = "https://huggingface.co/datasets/NomaDamas/Ko-StrategyQA/resolve/main/ko-strategy-qa_train.json"
    
    # 1. 원본 데이터 다운로드
    response = requests.get(url)
    data = response.json()
    
    # 2. 데이터 구조 확인 및 행(Row) 단위의 리스트로 변환
    records = []
    if isinstance(data, dict):
        # 데이터가 { "ID": { 세부정보 } } 형태일 경우
        for doc_id, doc_info in data.items():
            doc_info['id'] = doc_id # ID 값 보존
            records.append(doc_info)
    else:
        # 데이터가 이미 리스트 형태일 경우
        records = data
        
    # 3. PyArrow 에러의 주범(중첩 리스트/딕셔너리)을 모조리 JSON 문자열로 강제 변환
    for record in records:
        for key, value in record.items():
            if isinstance(value, (list, dict)):
                # 리스트나 딕셔너리는 문자열로 안전하게 묶어버립니다.
                record[key] = json.dumps(value, ensure_ascii=False)
                
    # 4. DataFrame 및 Dataset 변환
    df = pd.DataFrame(records)
    dataset = Dataset.from_pandas(df)
    
    print(f"✅ 데이터 로드 완료! (총 {len(dataset)}개)")
    return dataset

if __name__ == "__main__":
    print("📥 StrategyQA 로드 중...")
    # strategy_qa는 답변이 bool(True/False) 형태입니다.
    dataset = load_data()

    score = 0
    total = 5
    start_idx = 0

    for i, item in enumerate(dataset.select(range(start_idx, start_idx+total)), 1):
        question = item['question']
        correct_answer = "yes" if item['answer'] else "no"
        
        print(f"\n" + "="*80 + f"\n📝 [문제 {i}/{total}] {question}\n🔑 정답: {correct_answer}\n" + "="*80)
        
        success, info = adapt(question, "", 0, 3)
        
        final_answer = llm.invoke(f"정보: {info}\n질문: {question}\n답: yes or no?").content.strip().lower()
        
        print(f"\n🤖 제출: {final_answer}")
        if correct_answer in final_answer:
            print("🎉 정답!"); score += 1
        else: print("❌ 오답")
    
    print(f"\n🏆 최종: {score}/{total}")