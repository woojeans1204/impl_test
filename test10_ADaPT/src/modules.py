from langchain_core.messages import HumanMessage, SystemMessage
from src.tools import llm, base_executor_agent
from src.models import EvaluatorOutput, PlannerOutput

def run_executor(task: str, context: str) -> str:
    system_prompt = "당신은 지식 탐색 실행기입니다. 반드시 도구를 사용해 팩트를 확인하세요."
    prompt = f"컨텍스트: {context}\n임무: {task}"
    try:
        result = base_executor_agent.invoke({
            "messages": [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        }, {"recursion_limit": 10})
        return result["messages"][-1].content
    except Exception as e:
        return f"Executor Failure: {e}"

def run_evaluator(task: str, raw_output: str) -> EvaluatorOutput:
    eval_prompt = f"태스크: {task}\n결과물: {raw_output}\n\n위 결과가 태스크를 완벽히 해결했나요?"
    evaluator = llm.with_structured_output(EvaluatorOutput)
    result = evaluator.invoke(eval_prompt)
    if not result.is_success:
        print(f"    [⚠️ Evaluator 실패] 사유: {result.result_summary}")
    return result

def run_planner(task: str, fail_reason: str) -> PlannerOutput:
    prompt = f"태스크: '{task}' / 실패이유: '{fail_reason}'\n이 질문을 3단계 이하의 구체적인 팩트 체크 단계로 쪼개세요."
    planner = llm.with_structured_output(PlannerOutput)
    return planner.invoke(prompt)