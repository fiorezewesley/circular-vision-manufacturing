from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from app.config import MODELS_DIR, OUTPUTS_DIR
from app.data_loader import build_image_index
from app.feature_extraction import extract_features_from_path
from app.sustainability_score import build_sustainability_decision


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images_from_directory(image_dir: str | Path) -> pd.DataFrame:
    base = Path(image_dir)

    if not base.exists():
        raise FileNotFoundError(f"Image directory not found: {base}")

    rows = []

    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            rows.append(
                {
                    "image_path": str(path),
                    "source_label": path.parent.name,
                }
            )

    if not rows:
        raise ValueError("No valid images were found in the provided image directory.")

    return pd.DataFrame(rows)


def load_model_payload(model_path: str | Path) -> tuple[object, list[str]]:
    payload = joblib.load(model_path)

    if isinstance(payload, dict):
        model = payload.get("model")
        classes = payload.get("classes", [])

        if model is None:
            raise ValueError("Model payload does not contain a 'model' key.")

        classes = [str(item) for item in classes]
        return model, classes

    return payload, []


def normalize_prediction(prediction, classes: list[str]) -> str:
    if isinstance(prediction, (np.integer, int)):
        index = int(prediction)

        if classes and 0 <= index < len(classes):
            return str(classes[index])

        return str(index)

    return str(prediction)


def build_feature_matrix(image_paths: list[str]) -> tuple[np.ndarray, list[str]]:
    features = []
    valid_paths = []
    failed = []

    for image_path in tqdm(image_paths, desc="Extracting visual features"):
        try:
            vector = extract_features_from_path(image_path)
            features.append(vector)
            valid_paths.append(image_path)
        except Exception as exc:
            failed.append(
                {
                    "image_path": image_path,
                    "error": str(exc),
                }
            )

    if failed:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failed).to_csv(OUTPUTS_DIR / "batch_failed_images.csv", index=False)
        print(f"Skipped {len(failed)} unreadable images. Details saved to outputs/batch_failed_images.csv")

    if not features:
        raise ValueError("Feature extraction failed for all images.")

    return np.vstack(features), valid_paths


def generate_report(
    image_df: pd.DataFrame,
    model_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=UserWarning)

    model, classes = load_model_payload(model_path)
    image_paths = image_df["image_path"].tolist()

    x, valid_paths = build_feature_matrix(image_paths)

    raw_predictions = model.predict(x)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)

    rows = []

    for index, image_path in enumerate(valid_paths):
        predicted_class = normalize_prediction(raw_predictions[index], classes)

        confidence = None

        if probabilities is not None:
            confidence = float(np.max(probabilities[index]))

        decision = build_sustainability_decision(predicted_class)

        matched = image_df.loc[image_df["image_path"] == image_path]

        source_label = None

        if not matched.empty and "label" in matched.columns:
            source_label = matched.iloc[0]["label"]
        elif not matched.empty and "source_label" in matched.columns:
            source_label = matched.iloc[0]["source_label"]

        rows.append(
            {
                "image_path": image_path,
                "source_label": source_label,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "material_stream": decision["material_stream"],
                "recommended_action": decision["recommended_action"],
                "priority": decision["priority"],
                "circularity_score": decision["circularity_score"],
                "estimated_landfill_avoidance_kg": decision["estimated_landfill_avoidance_kg"],
                "special_handling": decision["special_handling"],
            }
        )

    report = pd.DataFrame(rows)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset root with class folders.",
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory with images for batch inference.",
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum number of images per class when using --dataset-dir.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "circular_vision_model.joblib"),
        help="Path to trained model.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR / "sustainability_report.csv"),
        help="Output CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_dir and not args.image_dir:
        raise ValueError("Provide either --dataset-dir or --image-dir.")

    if args.dataset_dir:
        image_df = build_image_index(args.dataset_dir, max_per_class=args.max_per_class)
    else:
        image_df = collect_images_from_directory(args.image_dir)

    report = generate_report(
        image_df=image_df,
        model_path=args.model,
        output_path=args.output,
    )

    print(f"Report generated: {args.output}")
    print(f"Processed images: {len(report)}")
    print(f"Average circularity score: {report['circularity_score'].mean():.2f}")
    print(f"Estimated avoided landfill: {report['estimated_landfill_avoidance_kg'].sum():.2f} kg")
    print(f"Special handling alerts: {int(report['special_handling'].sum())}")


if __name__ == "__main__":
    main()