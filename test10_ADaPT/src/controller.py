from src.modules import run_executor, run_evaluator, run_planner

def adapt(task: str, context: str, depth: int, max_depth: int = 3, root_task: str = None) -> tuple[bool, str]:
    if root_task is None:
        root_task = task
        
    indent = "  " * depth
    print(f"\n{indent}▶️ [Depth {depth}] ADaPT: {task}")
    if depth > max_depth: return False, "Max depth reached"

    raw_output = run_executor(root_task, task, context)
    eval_result = run_evaluator(root_task, task, raw_output)
    
    if eval_result.is_success:
        print(f"{indent}✅ 성공: {eval_result.result_summary[:50]}...")
        return True, eval_result.result_summary

    print(f"{indent}⚠️ 실패. Planner 분할 중...")
    plan = run_planner(root_task, task, eval_result.result_summary)
    print(f"{indent}📋 분할 [{plan.operator}]: {plan.sub_tasks}")

    if plan.operator == "AND":
        acc_context, combined = context, ""
        for sub in plan.sub_tasks:
            ok, res = adapt(sub, acc_context, depth + 1, max_depth, root_task)
            if not ok: return False, f"Fail at {sub}"
            acc_context += f"\n[{res}]"
            combined += f" {res}"
        return True, combined
    elif plan.operator == "OR":
        for sub in plan.sub_tasks:
            ok, res = adapt(sub, context, depth + 1, max_depth, root_task)
            if ok: return True, res
        return False, "All OR failed"
    return False, "Error"