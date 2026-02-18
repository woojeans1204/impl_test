현재까지 진행된 **Pre-training(사전 학습)** 단계에 대한 `README.md` 초안입니다.

프로젝트의 발전 과정(Shakespeare → TinyStories → FineWeb-Edu)과 현재 구축된 파일 시스템 구조를 명확하게 반영하여 작성했습니다. 이 파일을 프로젝트 최상단(`readme.md`)에 저장하시면 됩니다.

---

# 🧠 Custom NanoGPT: Pre-training Phase

이 프로젝트는 GPT(Generative Pre-trained Transformer) 모델을 바닥부터 구현하고, 단계별 데이터셋을 통해 사전 학습(Pre-training)을 수행한 기록입니다. **Andrej Karpathy의 NanoGPT**를 기반으로 하며, 다중 GPU 환경(`Accelerate`)과 대용량 데이터셋 처리에 최적화되어 있습니다.

## 📂 Project Structure

현재 프로젝트의 디렉토리 구조입니다.

```text
├── configs/               # 실험별 하이퍼파라미터 설정 (YAML)
│   └── exp260218/         # 2026-02-18 실험군
│       ├── base.yaml      # 기본 설정
│       ├── stories.yaml   # TinyStories용 설정
│       └── fineweb.yaml   # FineWeb-Edu용 설정
├── data/                  # 학습 데이터 및 전처리 스크립트
│   ├── shakespeare/       # (Step 1) 문자/단어 단위 기초 학습
│   ├── tinystories/       # (Step 2) 기초 문법 및 이야기 구조 학습
│   └── fineweb/           # (Step 3) 일반 상식 및 논리 학습 (Current)
├── results/               # 학습 결과물 (체크포인트, 로그, 샘플)
│   ├── shakespeare_gpt_v1
│   ├── tinystories_gpt_v1
│   └── fineweb_gpt_v2     # 현재 메인 실험 결과
├── src/                   # 핵심 소스 코드
│   ├── model.py           # GPT 모델 아키텍처 (PyTorch)
│   ├── trainer.py         # 학습 루프, 체크포인트, 샘플링 로직
│   └── dataset.py         # 대용량 데이터 로더 (Memory mapping)
├── train.py               # 단일 실험 실행 스크립트
├── run_manager.py         # 실험 스케줄러 (순차적 실험 실행)
├── inference.py           # 텍스트 생성(Inference) 스크립트
└── experiment_list.conf   # run_manager가 실행할 실험 목록

```

---

## 🚀 Quick Start

### 1. Dependencies

필요한 라이브러리를 설치합니다.

```bash
pip install torch numpy transformers datasets tiktoken accelerate pyyaml tqdm

```

### 2. Data Preparation

데이터셋 크기에 따라 단계별로 전처리를 수행합니다. 전처리 결과는 `.bin` (uint16) 형태로 저장됩니다.

**Step 1: Shakespeare (Char/Word level)**
가벼운 테스트용입니다.

```bash
python data/shakespeare/prepare_gptT.py

```

**Step 2: TinyStories (Narrative)**
문법과 기초적인 스토리텔링을 학습합니다.

```bash
python data/tinystories/prepare_tinystories.py

```

**Step 3: FineWeb-Edu (Knowledge & Reasoning)**
웹상의 고품질 교육 데이터를 학습합니다. (현재 메인)

```bash
python data/fineweb/prepare_fineweb.py

```

---

## ⚙️ Configuration

실험 설정은 `configs/` 폴더 내의 YAML 파일로 관리합니다.

| 설정 파일 | 용도 | 주요 특징 |
| --- | --- | --- |
| `stories.yaml` | TinyStories 학습 | 작은 모델, 빠른 수렴 확인용 |
| `fineweb.yaml` | FineWeb-Edu 학습 | **Main Model**. `n_layer=12`, `n_head=12`, `n_embd=768` (GPT-2 Small급) |

---

## 🔥 Training

### 단일 실험 실행 (`train.py`)

특정 설정 파일 하나로 학습을 시작합니다.

```bash
accelerate launch train.py --config configs/exp260218/fineweb.yaml

```

### 실험 스케줄러 실행 (`run_manager.py`)

여러 실험을 순차적으로 돌려야 할 때 사용합니다. `experiment_list.conf`에 등록된 YAML 파일들을 차례대로 실행합니다.

```bash
# 1. 실행할 리스트 확인
cat experiment_list.conf
# (예시 내용)
# stories.yaml
# fineweb.yaml

# 2. 매니저 실행
python run_manager.py

```

**Key Features:**

* **Accelerate Integration:** 단일 GPU 및 다중 GPU 환경 자동 대응.
* **Resume Capability:** 중단된 학습 시 `checkpoints/last.pth`를 감지하여 자동 재개.
* **Random Sampling:** 대용량 데이터셋(FineWeb)의 경우, 에포크마다 데이터를 랜덤하게 샘플링하여 효율적으로 학습.

---

## 🧪 Results & Monitoring

학습 결과는 `results/{실험명}/` 아래에 저장됩니다.

* **checkpoints/**: `last.pth` (최신), `ckpt_epoch_*.pth` (주기적 저장)
* **logs/**: TensorBoard 로그. `tensorboard --logdir results`로 확인 가능.
* **samples/**: 학습 중간에 생성된 텍스트 샘플.
* **Fixed Prompts:** 모델의 발전 과정을 비교하기 위해 매 에포크마다 4개의 고정된 질문("AI란 무엇인가", "1+1은" 등)에 대한 답변을 생성합니다.



**현재 진행 상황 (Example):**

* `shakespeare_gpt_v1`: 초기 구조 검증 완료.
* `fineweb_gpt_v2`: **100 Epoch 달성**. 문법 완성 및 기본 상식 추론 가능 단계.

---

## 💬 Inference

학습된 모델(`ckpt`)을 사용하여 텍스트를 생성합니다.

```bash
python inference.py

```

* `inference.py` 내부의 `CHECKPOINT_PATH`를 원하는 모델 경로(예: `results/fineweb_gpt_v2/checkpoints/last.pth`)로 수정하여 사용하세요.
* GPT-2 `tiktoken`을 사용하여 자연스러운 토크나이징을 지원합니다.

---

## 📝 Note

* **Hardware:** V100 GPU 환경에서 테스트되었습니다.
* **Dataset:** `fineweb/train.bin`은 용량 문제로 샘플링된 데이터(약 1%~10%)를 사용할 수 있습니다.
* **Next Step:** 현재 Pre-training이 완료되었으며, Instruction Tuning(Alpaca 데이터셋 등)을 통한 Fine-tuning 단계로 넘어갈 준비가 되었습니다.