import os
import base64
import io
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model once
MODEL_PATH = "emotiondetector.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Put your label order here (match your training order)
EMOTION_LABELS = ['angry','depression','disgust','fear','happy','neutral','sad','surprise'][:model.output_shape[-1]]

# Haar cascade for face detection (fallback to full image if none)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def _decode_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img

def _decode_image_from_base64(data_url: str) -> np.ndarray:
    # data:image/png;base64,AAAA...
    b64 = data_url.split(',', 1)[1] if ',' in data_url else data_url
    image_bytes = base64.b64decode(b64)
    return _decode_image_from_bytes(image_bytes)

def detect_and_preprocess(bgr_img: np.ndarray):
    """Return (input_tensor, bbox or None) for the biggest detected face or full image fallback."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))
    roi = None
    bbox = None

    if len(faces) > 0:
        # choose the largest face
        x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
        roi = gray[y:y+h, x:x+w]
        bbox = [int(x), int(y), int(w), int(h)]
    else:
        # fallback: use full grayscale frame
        roi = gray

    roi = cv2.resize(roi, (48,48), interpolation=cv2.INTER_AREA)
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=(0, -1))  # (1,48,48,1)
    return roi, bbox

def predict_from_bgr(bgr_img: np.ndarray):
    x, box = detect_and_preprocess(bgr_img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return {
        "label": EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else str(idx),
        "confidence": float(preds[idx]),
        "bbox": box
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_upload():
    file = request.files.get("image")
    if not file:
        return jsonify({"error":"no file"}), 400
    bgr = _decode_image_from_bytes(file.read())
    if bgr is None:
        return jsonify({"error":"decode_failed"}), 400
    result = predict_from_bgr(bgr)
    return jsonify(result)

@app.route("/predict-frame", methods=["POST"])
def predict_frame():
    data_url = request.json.get("frame")
    if not data_url:
        return jsonify({"error":"no_frame"}), 400
    bgr = _decode_image_from_base64(data_url)
    if bgr is None:
        return jsonify({"error":"decode_failed"}), 400
    result = predict_from_bgr(bgr)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
