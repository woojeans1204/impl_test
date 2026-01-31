from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "configs"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECK_DIR = ROOT_DIR / "checkpoints"

# 확인용 출력
if __name__ == "__main__":
    print(f"Project Root: {ROOT_DIR}")

def is_acc(model):
    return hasattr(model, 'module')