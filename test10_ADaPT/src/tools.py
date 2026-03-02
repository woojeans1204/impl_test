import concurrent.futures
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.utilities.python import PythonREPL
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# 전역 LLM 설정
import os
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
# 1. LLM 및 도구 초기화
print("API KEY:", os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

python_repl = PythonREPL()

@tool
def run_python_code(code: str) -> str:
    """파이썬 코드를 실행하여 복잡한 계산이나 논리 검증을 수행합니다."""
    try: return python_repl.run(code)
    except Exception as e: return f"Python Error: {e}"

@tool
def search_wikipedia(query: str) -> str:
    """위키백과에서 공식적인 데이터를 검색합니다."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    try: return wiki.run(query)
    except Exception as e: return f"Wiki Error: {e}"
import math
from langchain_core.tools import tool

@tool
def simple_calculator(expression: str) -> str:
    """고급 수학 수식을 계산합니다.
    사칙연산(+, -, *, /)과 거듭제곱(**)을 지원합니다.
    또한 파이썬 math 모듈의 모든 함수와 상수를 사용할 수 있습니다.
    예: '1225 ** (1/3)', 'math.sqrt(16)', 'math.log(100, 10)', 'math.pi * 2'
    """
    try:
        # 안전한 실행 환경 구축: 내장 함수는 막고 math 모듈의 기능만 허용
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names['math'] = math  # LLM이 'math.sqrt()' 형태로 호출할 수 있도록 모듈 자체도 허용
        
        # eval을 사용해 수식 계산
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        # 결과가 소수점일 경우, 너무 긴 부동소수점 오차를 방지하기 위해 약간의 포매팅 적용
        if isinstance(result, float):
            # 소수점 아래 10자리까지만 표현하여 자잘한 오차 제거 (예: 1.0749999999999997 -> 1.075)
            result = round(result, 10)
            # .0으로 끝나는 정수형 실수는 정수로 변환
            if result.is_integer():
                result = int(result)
                
        return str(result)
    except Exception as e:
        return f"계산 오류: {e}. 수식을 파이썬 문법에 맞게 수정하세요. (거듭제곱은 ^ 대신 ** 를 사용하세요.)"
    
# 기본 ReAct 에이전트 생성
tools = [search_wikipedia, simple_calculator]
tools = [simple_calculator]
base_executor_agent = create_react_agent(llm, tools)