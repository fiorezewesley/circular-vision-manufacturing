from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import RAW_DATA_DIR, EXPECTED_CLASSES
from app.data_loader import normalize_class_name


def score_directory(path: Path) -> int:
    if not path.is_dir():
        return 0
    children = [normalize_class_name(child.name) for child in path.iterdir() if child.is_dir()]
    return len(set(children).intersection(EXPECTED_CLASSES))


def find_best_root(base: Path = RAW_DATA_DIR) -> Path:
    candidates = [base] + [path for path in base.rglob("*") if path.is_dir()]
    ranked = sorted(((score_directory(path), path) for path in candidates), reverse=True)
    if not ranked or ranked[0][0] == 0:
        raise FileNotFoundError("Could not find dataset root with expected class folders.")
    return ranked[0][1]


if __name__ == "__main__":
    print(find_best_root())