from pathlib import Path
import cv2
import numpy as np
from skimage import color, feature
from .config import IMAGE_SIZE


def load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image


def extract_color_histogram(image: np.ndarray, bins: int = 24) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    features = []
    ranges = [(0, 180), (0, 256), (0, 256)]

    for channel, value_range in enumerate(ranges):
        hist = cv2.calcHist([hsv], [channel], None, [bins], value_range).flatten()
        hist = hist / (hist.sum() + 1e-8)
        features.extend(hist)

    return np.array(features, dtype=np.float32)


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    gray = color.rgb2gray(image)
    gray_uint = (gray * 255).astype(np.uint8)

    glcm = feature.graycomatrix(
        gray_uint,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2],
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    values = [feature.graycoprops(glcm, prop).flatten() for prop in props]

    return np.concatenate(values).astype(np.float32)


def extract_edge_features(image: np.ndarray) -> np.ndarray:
    gray = color.rgb2gray(image)
    edges = feature.canny(gray, sigma=1.4)

    edge_density = edges.mean()
    rows = np.array_split(edges, 4, axis=0)
    row_density = np.array([block.mean() for block in rows], dtype=np.float32)

    cols = np.array_split(edges, 4, axis=1)
    col_density = np.array([block.mean() for block in cols], dtype=np.float32)

    return np.concatenate([[edge_density], row_density, col_density]).astype(np.float32)


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    gray = color.rgb2gray(image)

    hog = feature.hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return hog.astype(np.float32)


def extract_features_from_image(image: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            extract_color_histogram(image),
            extract_texture_features(image),
            extract_edge_features(image),
            extract_hog_features(image),
        ]
    ).astype(np.float32)


def extract_features_from_path(image_path: str | Path) -> np.ndarray:
    image = load_image(image_path)
    return extract_features_from_image(image)