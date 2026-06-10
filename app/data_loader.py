from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CLASS_ALIASES, EXPECTED_CLASSES, RANDOM_STATE
from .feature_extraction import extract_features_from_path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_class_name(name: str) -> str:
    cleaned = name.strip().lower().replace(" ", "-").replace("_", "-")
    return CLASS_ALIASES.get(cleaned, cleaned)


def iter_image_files(dataset_dir: str | Path) -> Iterable[tuple[Path, str]]:
    base = Path(dataset_dir)

    if not base.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base}")

    for path in base.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        label = normalize_class_name(path.parent.name)

        if label in EXPECTED_CLASSES:
            yield path, label


def build_image_index(dataset_dir: str | Path, max_per_class: int | None = None) -> pd.DataFrame:
    rows = []

    for path, label in iter_image_files(dataset_dir):
        rows.append(
            {
                "image_path": str(path),
                "label": label,
            }
        )

    if not rows:
        raise ValueError("No valid class folders with images were found.")

    if max_per_class:
        random.seed(RANDOM_STATE)
        grouped_rows = {}

        for row in rows:
            grouped_rows.setdefault(row["label"], []).append(row)

        sampled_rows = []

        for label, items in grouped_rows.items():
            sample_size = min(len(items), max_per_class)
            sampled_rows.extend(random.sample(items, sample_size))

        rows = sampled_rows

    df = pd.DataFrame(rows, columns=["image_path", "label"])
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Indexed {len(df)} images.")
    print("Columns:", list(df.columns))
    print("Class distribution:")
    print(df["label"].value_counts().sort_index())

    return df


def build_feature_matrix(index_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_rows = []
    label_rows = []
    failed_rows = []

    records = index_df[["image_path", "label"]].to_dict("records")

    for record in tqdm(records, total=len(records), desc="Extracting visual features"):
        image_path = record["image_path"]
        label = record["label"]

        try:
            vector = extract_features_from_path(image_path)
            feature_rows.append(vector)
            label_rows.append(label)
        except Exception as exc:
            failed_rows.append(
                {
                    "image_path": image_path,
                    "label": label,
                    "error": str(exc),
                }
            )

    if failed_rows:
        Path("outputs").mkdir(exist_ok=True)
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_csv("outputs/failed_images.csv", index=False)
        print(f"Skipped {len(failed_rows)} unreadable images. Details saved to outputs/failed_images.csv")

    if not feature_rows:
        raise ValueError("Feature extraction failed for all images.")

    x = np.vstack(feature_rows)
    y = np.array(label_rows)

    if len(x) != len(y):
        raise ValueError(f"Inconsistent feature and label sizes: X={len(x)}, y={len(y)}")

    class_names = sorted(set(y))

    print(f"Feature matrix: {x.shape}")
    print(f"Labels: {y.shape}")
    print(f"Classes: {class_names}")

    return x, y, class_names


def split_sample_files(
    dataset_dir: str | Path,
    destination_dir: str | Path,
    samples_per_class: int = 5,
) -> None:
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    df = build_image_index(dataset_dir)
    random.seed(RANDOM_STATE)

    for label, group in df.groupby("label"):
        class_dir = destination / label
        class_dir.mkdir(parents=True, exist_ok=True)

        sample_paths = random.sample(
            list(group["image_path"]),
            min(samples_per_class, len(group)),
        )

        for path in sample_paths:
            source = Path(path)
            target = class_dir / source.name

            if not target.exists():
                target.write_bytes(source.read_bytes())