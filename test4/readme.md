네, **"이미지 싹 걷어내고 텍스트 생성에만 집중한"** `test4` 폴더를 위한 깔끔하고 전문적인 `README.md`입니다.

연구 인턴으로서 **"남들이 다 하는 라이브러리(`diffusers`) 딸깍이 아니라, 텍스트 데이터의 이산성(Discreteness)을 어떻게 연속적인 Diffusion 모델에 태웠는지 원리를 이해하고 구현했다"**는 점을 강조했습니다.

그대로 복사해서 `README.md` 파일에 넣으시면 됩니다.

---

```markdown
# Tiny Text Diffusion: Shakespeare from Scratch 📜

`diffusers`와 같은 고수준 라이브러리 없이, **PyTorch**와 **Accelerate**만으로 밑바닥부터(From Scratch) 구현한 **문자 단위(Character-level) 텍스트 확산 모델**입니다.

이미지 생성 모델(DDPM)의 연속적인 확산 프로세스를 **이산적인(Discrete) 텍스트 데이터**에 적용하기 위해, 임베딩 공간(Embedding Space)에서의 확산 및 1D Transformer 아키텍처를 직접 설계하고 구현했습니다.

---

## 🎯 프로젝트 핵심 (Key Features)

* **Text-Only Focus**: 불필요한 이미지 처리 코드를 모두 제거하고, 오직 텍스트 생성 원리 구현에만 집중한 경량화 프로젝트입니다.
* **Embedding Space Diffusion**:
    * 텍스트(Token)는 불연속적이어서 가우시안 노이즈를 직접 더할 수 없는 문제를 해결하기 위해, **학습 가능한 임베딩 레이어**를 통해 연속 공간으로 맵핑한 후 확산을 적용했습니다.
* **Custom Architecture**:
    * 2D U-Net 대신 시퀀스 데이터 처리에 특화된 **1D Transformer Encoder**를 Diffusion의 Backbone으로 직접 구현했습니다.
* **Resource Efficient**:
    * 셰익스피어 데이터셋(Tiny Shakespeare)을 사용하여, 단일 T4 GPU(Colab Free Tier) 환경에서도 30분 내에 유의미한 텍스트 생성이 가능하도록 최적화했습니다.

---

## 📂 폴더 구조 (Directory Structure)

```bash
test4/
├── configs/
│   └── text_shakespeare.yaml  # 실험 하이퍼파라미터 설정
├── src/
│   ├── dataset_text.py      # 셰익스피어 데이터 로더 (Character Tokenizer 포함)
│   ├── diffusion.py         # Gaussian Diffusion 수학적 로직 (Beta Schedule, q_sample)
│   ├── model_transformer.py # 노이즈 예측을 위한 1D Transformer 모델
│   └── trainer_text.py      # 학습 루프, 로깅, 텍스트 샘플링(Decoding) 로직
├── train.py                 # 메인 실행 스크립트
└── requirements.txt         # 의존성 패키지

```

---

## 🔬 구현 원리 (Methodology)

이 프로젝트는 **"어떻게 미분이 불가능한 텍스트에 노이즈를 섞을 것인가?"**에 대한 해답을 다음과 같이 구현했습니다.

1. **Embedding (Discrete  Continuous)**:
* 입력 문자의 정수 인덱스(Token ID)를 학습 가능한 `nn.Embedding`을 통해 실수 벡터로 변환합니다.


2. **Forward Process (Diffusion)**:
* 변환된 임베딩 벡터에 시간 에 따른 가우시안 노이즈를 주입합니다.
* 


3. **Reverse Process (Denoising)**:
* **Transformer Model**이 노이즈가 섞인 벡터 와 시간 를 입력받아, 주입된 노이즈 를 예측합니다.


4. **Rounding (Continuous  Discrete)**:
* 샘플링 과정에서 디노이징된 벡터와 임베딩 테이블 내의 모든 토큰 벡터 간의 거리를 계산(Euclidean Distance)하여, 가장 가까운 문자로 변환합니다.



---

## 🚀 실행 방법 (Quick Start)

### 1. 환경 설정

```bash
pip install -r requirements.txt

```

### 2. 학습 시작

별도의 데이터 다운로드가 필요 없습니다. 실행 시 셰익스피어 데이터셋을 자동으로 다운로드합니다.

```bash
python train.py --config configs/text_shakespeare.yaml

```

### 3. 결과 확인

학습이 진행되면서 `results/shakespeare_v1/samples/` 폴더에 에포크별로 생성된 텍스트 파일(`epoch_0010.txt`)이 저장됩니다.

* **초기**: 무의미한 문자열 나열 (`wh$a.. g@`)
* **중기**: 단어 형태 형성 및 띄어쓰기 학습 (`the king shall...`)
* **후기**: 셰익스피어 문체와 유사한 희곡 대본 생성

---

## 🛠 설정 가이드 (Configuration)

`configs/text_shakespeare.yaml`에서 모델 크기와 학습 설정을 조절할 수 있습니다.

```yaml
model:
  seq_len: 64    # 문맥 길이 (Context Window)
  dim: 128       # 임베딩 및 히든 사이즈
  depth: 6       # Transformer 레이어 깊이
  heads: 4       # Attention Head 개수

diffusion:
  timesteps: 1000 # Diffusion Step

train:
  batch_size: 64
  epochs: 200
  lr: 0.001

```

---

### 📝 Author Note

본 프로젝트는 기존의 '이미지 생성' 중심의 튜토리얼을 벗어나, 생성형 AI의 핵심 원리를 텍스트 도메인에서 밑바닥부터 검증하기 위해 작성되었습니다.

```

```