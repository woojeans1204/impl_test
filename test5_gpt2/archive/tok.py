import tiktoken

enc = tiktoken.get_encoding("gpt2")
text = " HuggingFace is making FineWeb-Edu dataset!"
text = input()
tokens = enc.encode(text)

# 토큰 사이에 '|'를 넣어서 시각화
visualized = "|".join([enc.decode([t]) for t in tokens])
print(f"토큰 분리 현황: \n{visualized}")