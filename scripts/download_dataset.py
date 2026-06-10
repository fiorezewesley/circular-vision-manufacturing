from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import shutil
import kagglehub
from app.config import DATASET_SLUG, RAW_DATA_DIR


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    target = RAW_DATA_DIR / "garbage-classification"
    if target.exists():
        print(f"Dataset already exists at {target}")
        return
    shutil.copytree(dataset_path, target)
    print(f"Dataset copied to {target}")


if __name__ == "__main__":
    main()