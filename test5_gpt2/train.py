import argparse
import yaml
import torch
import os
from pathlib import Path

# 우리가 만든 모듈들
from src.model import GPT, GPTConfig
from src.trainer import Trainer
from src.dataset import create_dataloader

def main():
    # 1. Argument Parsing (run_manager가 넘겨주는 --config를 받기 위함)
    parser = argparse.ArgumentParser(description="Train NanoGPT")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    # 2. Config 로드
    print(f">>> Loading Config: {args.config}")
    # run_manager가 상대 경로를 넘겨주므로 그대로 읽으면 됩니다.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 3. 시드 고정 (재현성)
    seed = config['system'].get('seed', 42) if 'system' in config else 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f">>> Experiment: {config.get('system', {}).get('experiment_name', 'NanoGPT')}")
    print(f">>> Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # -------------------------------------------------------------------------
    # [핵심] 여기서부터 NanoGPT 로직 (의존성 주입)
    # -------------------------------------------------------------------------

    # 4. 모델 초기화 (Model Init)
    # YAML의 model 섹션을 언패킹(**)하여 GPTConfig에 주입
    gpt_config = GPTConfig(**config['model']) 
    model = GPT(gpt_config)

    # 5. 데이터 로더 준비 (Data Loader)
    # config['data']['data_dir'] 경로에 .bin 파일이 있어야 함
    print(">>> Loading Data...")
    train_loader = create_dataloader(
        data_dir=config['data']['data_dir'],
        block_size=config['model']['block_size'],
        batch_size=config['train']['batch_size'],
        split='train'
    )
    # 검증용 로더 (선택 사항, 없으면 에러 처리 필요하지만 보통 만듦)
    val_loader = create_dataloader(
        data_dir=config['data']['data_dir'],
        block_size=config['model']['block_size'],
        batch_size=config['train']['batch_size'],
        split='val'
    )

    # 6. 트레이너 실행 (Trainer Run)
    # 모델, 로더, 설정 전체를 Trainer에게 넘김
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == '__main__':
    main()