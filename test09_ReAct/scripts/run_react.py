import os
import re
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def calculate(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def search_wikipedia(query):
    if "대통령" in query:
        return "한국의 현재 대통령 나이는 63세입니다."
    return "검색 결과가 없습니다."

def react_loop(question):
    prompt = """당신은 질문에 답하는 AI입니다. 다음 도구를 사용할 수 있습니다:
    1. calculate(expression): 수학 수식을 계산합니다.
    2. search_wikipedia(query): 위키백과에서 정보를 검색합니다.
    
    규칙: 반드시 다음 형식을 지켜서 출력하세요.
    Thought: [문제를 해결하기 위해 무엇을 해야 할지 생각]
    Action: [사용할 도구 이름(인자)] (예: calculate(3 * 5) 또는 search_wikipedia(대통령))
    Observation: [도구의 실행 결과]
    ...
    Final Answer: [최종 정답]
    """
    
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
    
    for step in range(5):
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, stop=["Observation:"])
        text = response.choices[0].message.content
        print(f"\n[LLM 응답]\n{text}")
        messages.append({"role": "assistant", "content": text})
        
        if "Final Answer:" in text:
            break
            
        calc_match = re.search(r"Action: calculate\((.*?)\)", text)
        wiki_match = re.search(r"Action: search_wikipedia\((.*?)\)", text)
        
        observation = ""
        if calc_match:
            observation = calculate(calc_match.group(1))
        elif wiki_match:
            observation = search_wikipedia(wiki_match.group(1))
            
        if observation:
            print(f"\n[도구 실행 결과]\nObservation: {observation}")
            messages.append({"role": "user", "content": f"Observation: {observation}"})

if __name__ == "__main__":
    react_loop("사과 5개와 바나나 3개의 총합에 한국의 현재 대통령 나이를 곱하면 얼마인가요?")