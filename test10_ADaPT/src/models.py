from typing import List, Literal
from pydantic import BaseModel, Field

class EvaluatorOutput(BaseModel):
    is_success: bool = Field(description="태스크가 성공적으로 완수되었는지 여부")
    result_summary: str = Field(description="성공/실패 이유 요약")

class PlannerOutput(BaseModel):
    operator: Literal["AND", "OR"] = Field(description="논리 관계")
    sub_tasks: List[str] = Field(description="구체적인 하위 작업 (최대 3개)")