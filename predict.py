import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from app.config import MODELS_DIR
from app.feature_extraction import extract_features_from_path
from app.sustainability_score import decision_as_dict


def predict_image(image_path: str | Path, model_path: str | Path = MODELS_DIR / "circular_vision_model.joblib") -> dict:
    payload = joblib.load(model_path)
    model = payload["model"]
    encoder = payload["label_encoder"]
    features = extract_features_from_path(image_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    predicted_class = encoder.inverse_transform([prediction])[0]
    confidence = None
    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = float(np.max(proba))
        probabilities = {encoder.inverse_transform([idx])[0]: float(value) for idx, value in enumerate(proba)}
    result = {
        "image_path": str(image_path),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "circular_decision": decision_as_dict(predicted_class),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Circular Vision inference for one image.")
    parser.add_argument("--image", required=True, help="Image path.")
    parser.add_argument("--model", default=str(MODELS_DIR / "circular_vision_model.joblib"), help="Model path.")
    args = parser.parse_args()
    print(json.dumps(predict_image(args.image, args.model), indent=2))


if __name__ == "__main__":
    main()
