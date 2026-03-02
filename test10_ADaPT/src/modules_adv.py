from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from src.tools import llm, base_executor_agent
from src.models import EvaluatorOutput
from src.modules import run_planner # 지우지 말 것

def run_executor_adv(root_task: str, current_task: str, context: str, verbose: bool = False) -> str:
    """실행자: 도구를 사용하여 현재 태스크를 해결합니다."""
    system_prompt = """[역할: 지식 탐색 및 정밀 실행기(Executor)]
당신은 주어진 목표를 해결하기 위해 도구를 사용하는 실행기입니다.
반드시 도구를 사용해 팩트를 확인하고, 명확한 답을 도출하십시오. 
특히 복잡한 계산의 경우 절대 암산하지 말고 반드시 계산기 도구를 사용하십시오."""
    
    prompt = f"[전체 문제]: {root_task}\n[이전 맥락]: {context}\n[현재 목표]: {current_task}"

    try:
        result = base_executor_agent.invoke({
            "messages": [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        }, {"recursion_limit": 15})
        
        full_trace = ""
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                if msg.content: full_trace += f"Thought: {msg.content}\n"
                if msg.tool_calls: full_trace += f"Action: {msg.tool_calls[0]['name']}({msg.tool_calls[0]['args']})\n"
            elif isinstance(msg, ToolMessage):
                full_trace += f"Observation: {msg.content}\n"
        
        full_trace = full_trace if full_trace else "결과 없음"
        if verbose:
            print(f"\n[Executor Trace]\n{full_trace}")
        return full_trace
    except Exception as e:
        return f"[ERROR] {str(e)}"

def run_critique_adv(root_task: str, current_task: str, executor_output: str, verbose: bool = False) -> str:
    """비판자: 실행자의 결과에서 논리적 결함이나 계산 실수를 찾아냅니다."""
    prompt = f"""[역할: 날카로운 비판자(Critique)]
당신의 역할은 실행자의 답변에서 잘못된 부분이나 치명적인 논리적 오류를 찾아 비판하는 것입니다.
주의: 실행자가 답을 내지 못했더라도 당신이 직접 답을 구하려고 하지 마십시오. 그것은 당신의 역할이 아닙니다. 오직 '비판'만 하십시오.

[전체 문제]: {root_task}
[현재 목표]: {current_task}
[실행 결과]: {executor_output}

위 실행 결과에서 논리적 결함이나 계산 실수를 찾아 비판하세요."""
    
    result = llm.invoke(prompt).content
    if verbose:
        print(f"\n😈 [비판자 결과]\n{result}")
    return result

def run_defender_adv(root_task: str, current_task: str, executor_output: str, critique_output: str, verbose: bool = False) -> str:
    """변호사: 비판자의 공격으로부터 실행자를 방어합니다."""
    prompt = f"""[역할: 실행자의 변호사(Defender)]
당신의 역할은 비판자의 억지 주장을 논리적, 수학적 팩트로 산산조각 내는 것입니다.

[변호 수칙 - 반드시 지킬 것]
1. 수학적 팩트 체크: 비판자가 "공식이 틀렸다"고 주장할 때, 만약 실행자의 공식이 수학적으로 올바른 변형(동치)이라면 "비판자는 수학적 지식이 부족하다"며 강력히 논파하십시오. 강력하게 논증하십시오.
2. 맹목적 옹호 금지: 실행자가 명백한 계산 실수를 했거나 도구를 쓰지 않고 엉뚱한 답을 냈다면 변호를 포기하고 인정하십시오.
3. 직접 풀이 금지: 변호 과정에서 새로운 정답을 직접 계산해 내지 마십시오.

[현재 목표]: {current_task}
[실행 결과]: {executor_output}
[비판 내용]: {critique_output}

비판자의 주장이 부당하거나 실행자가 목표를 달성했다면 실행자를 옹호하는 변론을 하세요."""
    
    result = llm.invoke(prompt).content
    if verbose:
        print(f"\n🛡️ [변호사 결과]\n{result}")
    return result

def run_evaluator_adv(root_task: str, current_task: str, raw_output: str, critique_output: str, defender_output: str, verbose: bool = False) -> EvaluatorOutput:
    """판사: 모든 의견을 듣고 현재 태스크의 성공 여부를 결정합니다."""
    eval_prompt = f"""[역할: 공정한 판사(Judge)]
당신의 역할은 모든 의견을 듣고 현재 태스크의 성공 여부를 결정하는 것입니다.
비판자의 답변과 변호사의 답변을 검토하여 성공 여부를 결정하세요. 

[현재 목표]: {current_task}
[실행 결과]: {raw_output}
[비판 내용]: {critique_output}
[변론 내용]: {defender_output}

[판결 지침]
0. 수학적 팩트 우선
1. 질문의 의도를 생각하고 맥락에 맞지 않거나 억지스러운 비판은 무시하십시오.
2. 실행자가 현재 목표를 완수하지 못한 경우에만 기각(is_success=False)하십시오.
3. 실행자가 현재 목표를 완수했고 정답이 확실하다면 승인(is_success=True)하십시오.
4. 승인할 경우, result_summary에는 오로지 '실행자(Executor)의 도출 과정과 결과'만 요약해서 적으십시오. 비판 내용이나 공방 과정은 요약에 절대 포함하지 마십시오.
5. 요약의 가장 마지막에는 반드시 '정답: [숫자]' 형식으로 답만 명확히 적어주십시오."""
    
    evaluator = llm.with_structured_output(EvaluatorOutput)
    result = evaluator.invoke(eval_prompt)
    if verbose:
        status = "✅ 승인" if result.is_success else "❌ 기각"
        print(f"\n⚖️ [판사 최종 판결] {status}\n요약: {result.result_summary}")
    return result