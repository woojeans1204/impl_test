import concurrent.futures
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.utilities.python import PythonREPL
from langgraph.prebuilt import create_react_agent

# 전역 LLM 설정
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

ddg_search = DuckDuckGoSearchRun()

@tool
def search_internet(query: str) -> str:
    """인터넷에서 최신 정보나 상식을 검색합니다."""
    def fetch(): return ddg_search.invoke(query)
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fetch)
            return future.result(timeout=10)
    except Exception as e: return f"Internet Search Error: {e}"

# 기본 ReAct 에이전트 생성
tools = [run_python_code, search_wikipedia, search_internet]
base_executor_agent = create_react_agent(llm, tools)