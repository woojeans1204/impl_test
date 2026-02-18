기존 Diffusion 프로젝트의 체계적인 실험 관리 구조(`run_manager.py`, `configs/` 등)를 그대로 살리면서, 코어 엔진만 **NanoGPT**로 교체한 형태의 `README.md`입니다.

`src/diffusion.py`는 제거하거나 `src/model.py`로 기능이 통합된 것으로 가정하고 작성했습니다.

conda create -n study python=3.10 -y
conda activate study
pip install -r requirements.txt

---

# NanoGPT V100 Framework

이 프로젝트는 기존의 실험 관리 프레임워크를 기반으로, **NVIDIA V100** 환경에서 **GPT(Decoder-only Transformer)** 모델을 바닥부터(Scratch) 학습시키기 위해 재구성되었습니다.

## ⚡ Key Features

* **Structure Inheritance:** 기존 Diffusion 프로젝트의 `run_manager`와 `YAML` 기반 설정 관리 시스템을 그대로 사용하여 실험의 재현성과 관리를 용이하게 함.
* **V100 Optimization:** PyTorch 2.0 `F.scaled_dot_product_attention` 및 `Mixed Precision (AMP)`을 적용하여 V100에서 학습 속도 극대화.
* **Custom Trainer:** 이미지 생성 로직을 텍스트 생성(Next Token Prediction) 로직으로 전면 교체.

## 📂 Project Structure

```text
.
├── configs/                 # 실험별 하이퍼파라미터 설정 (YAML)
│   ├── base.yaml            # 기본 설정
│   └── exp_v100/            # V100 최적화 실험군
├── data/                    # 텍스트 데이터셋 및 전처리 스크립트
│   └── tinystories/         # 예: TinyStories 데이터
├── results/                 # 실험 결과 (Logs, Checkpoints, Samples)
├── src/                     # 핵심 소스 코드
│   ├── __init__.py
│   ├── dataset.py           # 텍스트 데이터 로더 (np.memmap 기반)
│   ├── model.py             # GPT 아키텍처 (CausalSelfAttention, MLP)
│   ├── trainer.py           # 학습 루프 & 텍스트 생성 평가 로직
│   └── utils.py             # 시드 고정, 로깅 등 유틸리티
├── experiment_list.conf     # run_manager가 실행할 실험 목록
├── run_manager.py           # 실험 스케줄러 & 실행 관리자
├── train.py                 # 단일 실험 실행 진입점 (Entry Point)
└── .env                     # 환경 변수 (WandB API Key 등)

```

## 🚀 Getting Started

### 1. Environment Setup

```bash
# 가상 환경 생성 및 필수 패키지 설치
pip install torch numpy pyyaml tqdm wandb

```

### 2. Data Preparation

NanoGPT는 학습 속도를 위해 텍스트 데이터를 바이너리(`uint16`) 형태로 미리 전처리합니다.

```bash
cd data/tinystories
python prepare.py  # train.bin, val.bin 생성

```

### 3. Configuration (`configs/*.yaml`)

GPT 모델 사이즈와 학습 설정을 정의합니다. (기존 UNet 설정 대신 Transformer 설정 사용)

```yaml
model:
  n_layer: 6
  n_head: 6
  n_embd: 384
  block_size: 256
  dropout: 0.0

train:
  batch_size: 64
  learning_rate: 1e-3
  max_iters: 5000
  weight_decay: 0.1 # V100 학습 안정성 핵심

```

### 4. Training

#### 단일 실험 실행 (`train.py`)

```bash
python train.py --config configs/exp_v100/test_run.yaml

```

#### 실험 스케줄링 (`run_manager.py`)

여러 실험을 대기열에 걸어두고 순차적으로 실행합니다. `experiment_list.conf`에 실행할 config 경로를 적어주세요.

```bash
# experiment_list.conf 예시
# configs/exp_v100/exp1_layer6.yaml
# configs/exp_v100/exp2_layer12.yaml

python run_manager.py

```

## 🧠 Model Architecture

이 프로젝트의 `src/model.py`는 다음과 같은 최신 GPT 트렌드를 따릅니다:

* **Pre-LayerNorm:** Residual Connection 이전에 Norm을 적용하여 Deep Layer 학습 안정화.
* **GELU Activation:** ReLU 대신 GELU 사용.
* **Flash Attention:** `torch.nn.functional.scaled_dot_product_attention`을 사용하여 메모리 사용량 감소 및 속도 향상.

## 📊 Results & Monitoring

학습 결과는 `results/{experiment_name}/` 경로에 저장됩니다.

* **logs/**: Tensorboard 또는 WandB 로그.
* **checkpoints/**: `.pth` 모델 가중치 파일.
* **samples/**: 학습 중간에 생성된 텍스트 샘플 (생성 능력 확인용).

---

*Based on nanoGPT, Adapted for V100 Experiments.*

---

**어떠신가요?**
기존에 쓰시던 `run_manager.py`나 `experiment_list.conf` 같은 파일명을 그대로 명시해서, 님이 "아, 이거 원래 쓰던 그 파일이구나" 하고 바로 감을 잡을 수 있게 작성했습니다.

**다음으로 무엇을 도와드릴까요?**

1. 이 구조에 맞춰서 **`src/dataset.py` (텍스트 전용 로더)** 코드를 짜 드릴까요?
2. **`src/trainer.py`**에서 이미지가 아니라 **텍스트를 생성해서 저장하는 로직**을 구현해 드릴까요?
3. 준비되셨다면, 새로운 리드미를 덮어쓰고 보랏빛 지능을 깨우기 위해 **"아오!!!!"**를 외치시겠습니까? 🥵🟣