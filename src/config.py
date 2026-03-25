from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
