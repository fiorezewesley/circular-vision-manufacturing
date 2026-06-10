import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE
from app.data_loader import build_feature_matrix, build_image_index


def train_model(dataset_dir: str, max_per_class: int | None) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    index_df = build_image_index(dataset_dir, max_per_class=max_per_class)
    index_df.to_csv(PROCESSED_DATA_DIR / "image_index.csv", index=False)

    x, labels, class_names = build_feature_matrix(index_df)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "knn_baseline": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        ]),
        "random_forest": Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
        ]),
    }

    results = {}
    best_name = None
    best_score = -1.0
    best_model = None

    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")
        results[name] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "classification_report": classification_report(y_test, pred, target_names=encoder.classes_, output_dict=True),
        }
        if macro_f1 > best_score:
            best_name = name
            best_score = macro_f1
            best_model = model

    payload = {
        "model": best_model,
        "label_encoder": encoder,
        "classes": list(encoder.classes_),
        "feature_version": "color_texture_edge_hog_v1",
    }
    joblib.dump(payload, MODELS_DIR / "circular_vision_model.joblib")

    y_pred = best_model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(ax=ax, xticks_rotation=45, values_format="d")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    metrics = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "max_per_class": max_per_class,
        "total_images": int(len(index_df)),
        "classes": class_names,
        "best_model": best_name,
        "results": results,
    }
    with open(OUTPUTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    class_distribution = index_df["label"].value_counts().sort_index()
    class_distribution.to_csv(OUTPUTS_DIR / "class_distribution.csv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Circular Vision material classification models.")
    parser.add_argument("--dataset-dir", required=True, help="Path to the dataset root containing class folders.")
    parser.add_argument("--max-per-class", type=int, default=400, help="Maximum images per class for a fast balanced PoC.")
    args = parser.parse_args()
    metrics = train_model(args.dataset_dir, args.max_per_class)
    print(json.dumps({"best_model": metrics["best_model"], "total_images": metrics["total_images"]}, indent=2))


if __name__ == "__main__":
    main()
