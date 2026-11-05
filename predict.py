#!/usr/bin/env python3
"""
AgriDrone — Crop Health Prediction Module
=========================================
Standalone prediction module that loads the trained crop health model
and classifies aerial / drone crop images.

**Smart Fallback**: When no trained model (.h5 / .tflite) is available,
the module uses a heuristic image-analysis pipeline based on colour-channel
ratios (green dominance, brown ratio, yellow ratio) and texture variance
to produce realistic disease probability estimates.

Usage:
  # As a library
  from predict import predict_crop_health
  results = predict_crop_health("image.jpg")
  # [{"label": "Healthy", "score": 0.92}, ...]

  # CLI
  python predict.py path/to/image.jpg
"""

import os
import json
import sys
from pathlib import Path
from typing import Union

import numpy as np

# Suppress noisy TF logs when imported as a library
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PKL_PATH = BASE_DIR / "crop_health_model.pkl"
MODEL_H5_PATH = BASE_DIR / "crop_health_model.h5"
MODEL_TFLITE_PATH = BASE_DIR / "crop_health_model.tflite"
CLASS_LABELS_PATH = BASE_DIR / "class_labels.json"

IMG_SIZE = 224

# Default labels matching the training script's target classes
DEFAULT_LABELS = [
    "Apple - Black Rot",
    "Corn - Common Rust",
    "Corn - Northern Leaf Blight",
    "Grape - Black Rot",
    "Grape - Leaf Blight",
    "Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Yellow Leaf Curl Virus",
]


# ──────────────────────────────────────────────
# Label Loader
# ──────────────────────────────────────────────
def _load_labels() -> list[str]:
    """Load class labels from JSON, or return defaults."""
    if CLASS_LABELS_PATH.exists():
        try:
            with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if isinstance(labels, list) and len(labels) > 0:
                return labels
        except Exception:
            pass
    return DEFAULT_LABELS


# ──────────────────────────────────────────────
# Image Loader
# ──────────────────────────────────────────────
def _load_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Accept a file path (str / Path) or a numpy array.
    Returns a (IMG_SIZE, IMG_SIZE, 3) float32 array with pixels in [0, 255].
    """
    if isinstance(image_input, np.ndarray):
        img = image_input.copy()
        # Handle grayscale
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # drop alpha
        # Resize if needed
        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(img.astype(np.uint8))
                pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
                img = np.array(pil_img, dtype=np.float32)
            except ImportError:
                import cv2
                img = cv2.resize(img.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32)

    # File path
    path = Path(image_input)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        from PIL import Image as PILImage
        pil_img = PILImage.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return np.array(pil_img, dtype=np.float32)
    except ImportError:
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        return resized.astype(np.float32)


# ══════════════════════════════════════════════
#  MODEL-BASED PREDICTION
# ══════════════════════════════════════════════

def _predict_with_keras(image: np.ndarray, labels: list[str]) -> list[dict]:
    """Run inference with the full Keras .h5 model."""
    import tensorflow as tf

    model = tf.keras.models.load_model(str(MODEL_H5_PATH))
    # MobileNetV2 preprocessing: scale to [-1, 1]
    batch = np.expand_dims(image, axis=0)
    preds = model.predict(batch, verbose=0)[0]

    results = [
        {"label": labels[i], "score": round(float(preds[i]), 4)}
        for i in range(len(labels))
    ]
    return sorted(results, key=lambda x: x["score"], reverse=True)


def _predict_with_tflite(image: np.ndarray, labels: list[str]) -> list[dict]:
    """Run inference with the TFLite model (edge/embedded)."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input — match expected dtype
    input_dtype = input_details[0]["dtype"]
    batch = np.expand_dims(image, axis=0).astype(input_dtype)

    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    results = [
        {"label": labels[i], "score": round(float(preds[i]), 4)}
        for i in range(len(labels))
    ]
    return sorted(results, key=lambda x: x["score"], reverse=True)


# ══════════════════════════════════════════════
#  HEURISTIC FALLBACK (no model required)
# ══════════════════════════════════════════════

def _heuristic_predict(image: np.ndarray, labels: list[str]) -> list[dict]:
    """
    Analyse the image's pixel-level colour and texture features to produce
    realistic disease probability estimates — no ML model needed.

    Feature extraction:
      1. Green Channel Dominance (NDVI-style green ratio) — healthy indicator
      2. Brown Ratio (R-high, G-mid, B-low patches) — blight / rot indicator
      3. Yellow Ratio (R-high, G-high, B-low patches) — virus / curl indicator
      4. Texture Variance (Laplacian or std-dev) — spot / lesion detection
      5. Dark Spot Ratio — necrosis / mold indicator
    """
    # Ensure float [0, 255]
    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img *= 255.0

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    total = R + G + B + 1e-6  # avoid division by zero

    # ── Feature 1: Green Dominance ──────────────────
    # Fraction of pixels where green is the dominant channel
    green_dominant_mask = (G > R) & (G > B)
    green_ratio = float(np.mean(green_dominant_mask))

    # Normalised green intensity
    green_intensity = float(np.mean(G / total))

    # Combined green health score [0, 1]
    green_score = 0.6 * green_ratio + 0.4 * green_intensity * 3.0
    green_score = min(max(green_score, 0.0), 1.0)

    # ── Feature 2: Brown Ratio ──────────────────────
    # Brown ≈ R > 100, G in [50, 180], B < 100, R > G > B
    brown_mask = (
        (R > 90) & (G > 40) & (G < 190) & (B < 110) &
        (R > G) & (G > B) &
        ((R - B) > 30)
    )
    brown_ratio = float(np.mean(brown_mask))

    # ── Feature 3: Yellow Ratio ─────────────────────
    # Yellow ≈ R high, G high, B low
    yellow_mask = (
        (R > 140) & (G > 140) & (B < 120) &
        (np.abs(R.astype(np.float32) - G.astype(np.float32)) < 60)
    )
    yellow_ratio = float(np.mean(yellow_mask))

    # ── Feature 4: Texture Variance ─────────────────
    # High variance → spots / lesions / non-uniform
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    # Simple Laplacian approximation (3×3)
    from scipy.ndimage import laplace as _laplace_filter
    try:
        lap = _laplace_filter(gray)
        texture_var = float(np.var(lap)) / 10000.0  # normalise
    except Exception:
        # Fallback without scipy
        texture_var = float(np.std(gray)) / 80.0
    texture_var = min(texture_var, 1.0)

    # ── Feature 5: Dark Spot Ratio ──────────────────
    dark_mask = (gray < 60)
    dark_ratio = float(np.mean(dark_mask))

    # ── Feature 6: Pale / Chlorotic Ratio ───────────
    pale_mask = (G > 150) & (R > 150) & (B > 100) & ((G - B) < 60)
    pale_ratio = float(np.mean(pale_mask))

    # ══════════════════════════════════════════════
    # Map features → disease probabilities
    # ══════════════════════════════════════════════
    # Build raw affinity scores for each label
    scores: dict[str, float] = {}

    disease_signal = brown_ratio + yellow_ratio + dark_ratio * 0.5
    health_signal = green_score

    for label in labels:
        lbl = label.lower()

        if label == "Healthy":
            # High green, low brown / yellow / spots
            s = (
                green_score * 3.5
                - brown_ratio * 4.0
                - yellow_ratio * 3.0
                - texture_var * 1.5
                - dark_ratio * 2.0
                + 0.3  # base prior — many images are healthy
            )
            scores[label] = max(s, 0.05)

        elif "late blight" in lbl:
            # Dark brown/black lesions, high texture variance
            s = (
                brown_ratio * 2.5
                + dark_ratio * 3.0
                + texture_var * 2.0
                - green_score * 1.5
                + 0.1
            )
            scores[label] = max(s, 0.02)

        elif "early blight" in lbl:
            # Concentric brown rings — moderate brown, moderate texture
            s = (
                brown_ratio * 3.0
                + texture_var * 1.5
                + yellow_ratio * 0.8
                - green_score * 1.0
                + 0.08
            )
            scores[label] = max(s, 0.02)

        elif "leaf mold" in lbl:
            # Pale green/yellow patches on leaves, moderate dark
            s = (
                pale_ratio * 2.5
                + yellow_ratio * 1.5
                + dark_ratio * 1.0
                - green_score * 0.5
                + 0.05
            )
            scores[label] = max(s, 0.01)

        elif "yellow leaf curl" in lbl:
            # Strong yellow, curling (texture), low green
            s = (
                yellow_ratio * 4.0
                + texture_var * 1.5
                + pale_ratio * 1.0
                - green_score * 2.0
                + 0.05
            )
            scores[label] = max(s, 0.01)

        elif "common rust" in lbl:
            # Reddish-brown pustules — high brown, reddish tint
            red_ratio = float(np.mean((R > 150) & (G < 120) & (B < 100)))
            s = (
                brown_ratio * 2.0
                + red_ratio * 3.5
                + texture_var * 1.5
                - green_score * 1.0
                + 0.05
            )
            scores[label] = max(s, 0.01)

        elif "northern leaf blight" in lbl:
            # Long grey-green lesions — moderate brown, high texture
            s = (
                brown_ratio * 1.5
                + texture_var * 2.5
                + pale_ratio * 1.0
                - green_score * 0.8
                + 0.05
            )
            scores[label] = max(s, 0.01)

        elif "black rot" in lbl:
            # Dark spots / necrotic areas
            s = (
                dark_ratio * 4.0
                + brown_ratio * 2.0
                + texture_var * 1.5
                - green_score * 1.5
                + 0.05
            )
            scores[label] = max(s, 0.01)

        elif "leaf blight" in lbl:
            # General blight — brown + texture
            s = (
                brown_ratio * 2.5
                + texture_var * 2.0
                + dark_ratio * 1.0
                - green_score * 1.0
                + 0.05
            )
            scores[label] = max(s, 0.01)

        else:
            # Unknown label — small uniform score
            scores[label] = 0.02 + disease_signal * 0.5

    # ── Normalise to probabilities ──────────────────
    total_score = sum(scores.values())
    if total_score <= 0:
        total_score = 1.0

    results = [
        {"label": label, "score": round(scores[label] / total_score, 4)}
        for label in labels
    ]
    return sorted(results, key=lambda x: x["score"], reverse=True)


def _predict_with_pkl(image: np.ndarray, labels: list[str]) -> list[dict]:
    """Run inference with the trained scikit-learn aerial ensemble model."""
    import joblib
    from train_model import extract_features_from_array
    model = joblib.load(str(MODEL_PKL_PATH))
    feats = extract_features_from_array(image).reshape(1, -1)
    probs = model.predict_proba(feats)[0]

    model_classes = getattr(model, "classes_", list(range(len(labels))))
    results = []
    for idx, prob in enumerate(probs):
        class_target = model_classes[idx]
        if isinstance(class_target, (int, np.integer)) and class_target < len(labels):
            lbl = labels[class_target]
        else:
            lbl = str(class_target)
        results.append({"label": lbl, "score": round(float(prob), 4)})

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ══════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════

def predict_crop_health(
    image_input: Union[str, Path, np.ndarray],
    prefer_tflite: bool = False,
) -> list[dict]:
    """
    Predict crop health from an image.

    Args:
        image_input: File path (str/Path) or numpy array (H, W, 3).
        prefer_tflite: If True and the .tflite model exists, use it
                       instead of the full Keras model.

    Returns:
        Sorted list of dicts: [{"label": str, "score": float}, ...]
        Scores sum to ~1.0, descending by confidence.
    """
    labels = _load_labels()
    image = _load_image(image_input)

    # ── Try trained model first ─────────────────────
    if MODEL_PKL_PATH.exists():
        try:
            return _predict_with_pkl(image, labels)
        except Exception as e:
            print(f"⚠️  PKL model inference failed ({e}), trying alternatives…")

    if prefer_tflite and MODEL_TFLITE_PATH.exists():
        try:
            return _predict_with_tflite(image, labels)
        except Exception as e:
            print(f"⚠️  TFLite inference failed ({e}), trying .h5 fallback…")

    if MODEL_H5_PATH.exists():
        try:
            return _predict_with_keras(image, labels)
        except Exception as e:
            print(f"⚠️  Keras inference failed ({e}), trying TFLite fallback…")

    if MODEL_TFLITE_PATH.exists():
        try:
            return _predict_with_tflite(image, labels)
        except Exception as e:
            print(f"⚠️  TFLite inference failed ({e}), using heuristic fallback…")

    # ── Heuristic fallback ──────────────────────────
    return _heuristic_predict(image, labels)


def get_prediction_method() -> str:
    """Return which prediction backend will be used."""
    if MODEL_PKL_PATH.exists():
        return "ensemble_pkl"
    elif MODEL_H5_PATH.exists():
        return "keras_h5"
    elif MODEL_TFLITE_PATH.exists():
        return "tflite"
    else:
        return "heuristic_fallback"


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [--tflite]")
        print("\nPrediction backend: " + get_prediction_method())
        sys.exit(0)

    image_path = sys.argv[1]
    use_tflite = "--tflite" in sys.argv

    print(f"🔍 Analysing: {image_path}")
    print(f"   Backend : {get_prediction_method()}")
    print()

    results = predict_crop_health(image_path, prefer_tflite=use_tflite)

    print("📊 Crop Health Analysis Results:")
    print("─" * 45)
    for i, r in enumerate(results):
        bar = "█" * int(r["score"] * 30)
        marker = " ◀ TOP" if i == 0 else ""
        print(f"  {r['label']:40s} {r['score']*100:5.1f}%  {bar}{marker}")
    print("─" * 45)
    print(f"\n🏷️  Primary detection: {results[0]['label']} "
          f"({results[0]['score']*100:.1f}% confidence)")


if __name__ == "__main__":
    main()
