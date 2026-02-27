import argparse
import yaml
import torch
from src.trainer import Trainer
from src.__init__ import CONFIG_DIR

def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Model")
    parser.add_argument("--config", type=str, default=CONFIG_DIR/"mnist.yml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print(f">>> Starting experiment: {config['experiment_name']}")
    print(f">>> Using Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()