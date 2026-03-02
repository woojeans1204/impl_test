from langchain_core.messages import HumanMessage, SystemMessage
from src.tools import llm, base_executor_agent
from src.models import EvaluatorOutput, PlannerOutput

def run_executor(root_task: str, current_task: str, context: str) -> str:
    system_prompt = "당신은 지식 탐색 실행기입니다. 반드시 도구를 사용해 팩트를 확인하고 답을 내시오. 질문자의 의도를 반드시 파악하십시오."
    prompt = f"전체 원본 문제: {root_task}\n\n이전 컨텍스트: {context}\n\n현재 임무: {current_task}"
    try:
        result = base_executor_agent.invoke({
            "messages": [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        }, {"recursion_limit": 15})
        return result["messages"][-1].content
    except Exception as e:
        return f"Executor Failure: {e}"

def run_evaluator(root_task: str, current_task: str, raw_output: str) -> EvaluatorOutput:
    eval_prompt = f"""전체 원본 문제: {root_task}
현재 쪼개진 태스크: {current_task}
실행자 결과물: {raw_output}

위 결과물이 '현재 쪼개진 태스크'를 논리적으로 해결했고, '전체 원본 문제'를 해결하는 데 올바르게 기여했나요? 
현재 태스크의 목표를 달성했다면 성공으로 간주하세요."""
    evaluator = llm.with_structured_output(EvaluatorOutput)
    result = evaluator.invoke(eval_prompt)
    if not result.is_success:
        print(f"    [⚠️ Evaluator 실패] 사유: {result.result_summary}")
    return result

def run_planner(root_task: str, current_task: str, fail_reason: str) -> PlannerOutput:
    prompt = f"""전체 원본 문제: {root_task}
현재 실패한 태스크: '{current_task}' 
실패 이유: '{fail_reason}'

위 상황을 해결하기 위해 문제를 더 작게 쪼개세요.
주의: 만약 실패 이유가 '계산 과정 누락'이라면, 하위 태스크에 "구체적인 수식을 작성하거나 파이썬 도구를 사용하여 계산하라"는 지침을 명시적으로 포함해야 합니다. 
이전과 똑같은 방식으로 쪼개는 것을 반복하지 마세요."""
    planner = llm.with_structured_output(PlannerOutput)
    return planner.invoke(prompt)