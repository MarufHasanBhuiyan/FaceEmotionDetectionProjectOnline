import io
import base64
import time
from typing import List

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "emotiondetector.h5"
LABELS: List[str] = ["angry","depression","disgust","fear","happy","neutral","sad","surprise"]
IMG_SIZE = (48, 48)  # matches your training
THROTTLE_MS = 150     # server hint; client respects this to avoid spamming

# ----------------------------
# App & Model
# ----------------------------
app = Flask(__name__)

# Lazy import TensorFlow to make cold starts snappier
@app.before_first_request
def load_model_once():
    global model, tf, keras
    import tensorflow as tf
    from tensorflow import keras
    try:
        model = keras.models.load_model(MODEL_PATH)
        # Warm up (first call is slow).
        _ = model.predict(np.zeros((1, *IMG_SIZE, 1), dtype=np.float32))
        app.logger.info("Model loaded successfully.")
    except Exception as e:
        app.logger.exception(f"Failed to load model: {e}")
        raise

# ----------------------------
# Helpers
# ----------------------------
def preprocess_pil_to_tensor(pil_img: Image.Image) -> np.ndarray:
    """
    Convert an arbitrary PIL image to the model's expected input:
    - convert to grayscale
    - resize to 48x48 with high-quality resampling
    - to float32 in [0,1]
    - shape (1, 48, 48, 1)
    """
    # Convert to L (grayscale)
    img = pil_img.convert("L")
    # Letterbox to square first (avoid aspect distortion), then resize
    img = ImageOps.fit(img, IMG_SIZE, method=Image.Resampling.LANCZOS, bleed=0.0, centering=(0.5, 0.5))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # (1, H, W, 1)
    return arr

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def predict_probs(img_arr: np.ndarray) -> List[dict]:
    """
    Runs the model and returns a sorted list of class probabilities.
    """
    logits_or_probs = model.predict(img_arr, verbose=0)
    # If the model already ends with softmax, this is probs; else convert:
    if logits_or_probs.min() < 0 or logits_or_probs.max() > 1.0:
        probs = softmax(logits_or_probs)
    else:
        probs = logits_or_probs

    probs = probs[0].tolist()
    results = [{"label": LABELS[i], "prob": float(p)} for i, p in enumerate(probs)]
    results.sort(key=lambda d: d["prob"], reverse=True)
    return results

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html", labels=LABELS, throttle_ms=THROTTLE_MS)

@app.route("/healthz")
def health():
    return "ok", 200

@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    """
    Multipart form-data: file=@image.jpg
    Returns top class + full distribution.
    """
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    try:
        pil_img = Image.open(file.stream).convert("RGB")
        tensor = preprocess_pil_to_tensor(pil_img)
        results = predict_probs(tensor)
        return jsonify({
            "top": results[0],
            "results": results
        })
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """
    JSON: { "image": "data:image/jpeg;base64,...." }
    (Sent from the browser webcam.)
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "no image"}), 400
    b64 = data["image"]
    try:
        # strip header if present
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = preprocess_pil_to_tensor(pil_img)
        results = predict_probs(tensor)
        return jsonify({
            "top": results[0],
            "results": results
        })
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # local dev
    app.run(host="0.0.0.0", port=5000, debug=False)
