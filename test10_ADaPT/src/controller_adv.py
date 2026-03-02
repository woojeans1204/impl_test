from src.modules_adv import run_executor_adv, run_critique_adv, run_defender_adv, run_evaluator_adv, run_planner

def adapt_adv(task: str, context: str, depth: int, max_depth: int = 3, root_task: str = None) -> tuple[bool, str]:
    if root_task is None: root_task = task
    indent = "  " * depth  # <--- 이 부분이 빠져서 에러가 났었습니다.
    
    if depth > max_depth: return False, "최대 깊이 도달"

    print(f"\n{indent}▶️ [Depth {depth}] 태스크: {task}")
    verbose = True
    # 1. 실행 -> 비판 -> 변호 -> 판결
    raw = run_executor_adv(root_task, task, context, verbose)
    critique = run_critique_adv(root_task, task, raw, verbose)
    defender = run_defender_adv(root_task, task, raw, critique, verbose)
    eval_result = run_evaluator_adv(root_task, task, raw, critique, defender, verbose)

    if eval_result.is_success:
        print(f"{indent}✅ 성공: {eval_result.result_summary}")
        return True, eval_result.result_summary

    # 2. 실패 시 분할
    print(f"{indent}❌ 기각: {eval_result.result_summary}")
    plan = run_planner(root_task, task, eval_result.result_summary)
    print(f"{indent}📋 분할 [{plan.operator}]: {plan.sub_tasks}")

    if plan.operator == "AND":
        acc_context, last_res = context, ""
        for sub in plan.sub_tasks:
            ok, res = adapt_adv(sub, acc_context, depth + 1, max_depth, root_task)
            if not ok: return False, res
            acc_context += f"\n[알아낸 정보: {res}]"
            last_res = res # 마지막 단계의 결과가 최종 정답일 확률이 높음
        return True, last_res # AND의 경우 마지막 단계의 요약을 반환하여 정답 추출을 돕습니다.
        
    elif plan.operator == "OR":
        for sub in plan.sub_tasks:
            ok, res = adapt_adv(sub, context, depth + 1, max_depth, root_task)
            if ok: return True, res
        return False, "모든 시도 실패"
        
    return False, "에러 발생"