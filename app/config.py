from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = (128, 128)
RANDOM_STATE = 42
TEST_SIZE = 0.25

EXPECTED_CLASSES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]

CLASS_ALIASES = {
    "brown_glass": "brown-glass",
    "green_glass": "green-glass",
    "white_glass": "white-glass",
    "glass-brown": "brown-glass",
    "glass-green": "green-glass",
    "glass-white": "white-glass",
}

DATASET_SLUG = "mostafaabla/garbage-classification"
