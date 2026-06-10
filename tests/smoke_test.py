from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import tempfile
from PIL import Image
import numpy as np
from app.feature_extraction import extract_features_from_path
from app.sustainability_score import decision_as_dict


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "sample.jpg"
        image = Image.fromarray(np.full((128, 128, 3), [120, 90, 50], dtype=np.uint8))
        image.save(image_path)
        features = extract_features_from_path(image_path)
        decision = decision_as_dict("cardboard")
        assert features.ndim == 1
        assert features.size > 0
        assert decision["circularity_score"] == 90
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
