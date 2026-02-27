from datasets import load_dataset

def fetch_hotpot_qa():
    # 1. Hugging Face 허브에서 HotpotQA 데이터셋 로드
    print("📥 HotpotQA 데이터셋을 로드하는 중... (최초 1회 다운로드 시간이 소요됩니다.)")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation[:3]")

    print("\n" + "="*50)
    print("🔥 HotpotQA 샘플 데이터 3개 확인")
    print("="*50)

    # 2. 데이터셋 구조 파악을 위한 출력 루프
    for i, item in enumerate(dataset, 1):
        print(f"\n[샘플 {i}]")
        print(f"Q (질문): {item['question']}")
        print(f"A (정답): {item['answer']}")
        print(f"유형: {item['type']} | 난이도: {item['level']}")
        print("-" * 50)

if __name__ == "__main__":
    fetch_hotpot_qa()