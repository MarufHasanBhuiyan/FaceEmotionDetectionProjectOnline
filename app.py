import io
import time
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --------------------- Config ---------------------
LABELS = ['angry','depression','disgust','fear','happy','neutral','sad','surprise']

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
MODEL = load_model('emotiondetector.h5')
_ = MODEL.predict(np.zeros((1,48,48,1), dtype=np.float32))  # warmup

app = Flask(__name__)

# --------------------- Helpers ---------------------
def preprocess_face(gray_image, box):
    x, y, w, h = box
    face = gray_image[y:y+h, x:x+w]
    if face.size == 0:
        return None
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(img_to_array(face), axis=0)  # (1,48,48,1)
    return face

def decode_data_url_image(data_url):
    header, b64data = data_url.split(',', 1)
    img_bytes = base64.b64decode(b64data)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def pick_largest_face(faces):
    if len(faces) == 0:
        return None
    areas = [(w*h, (x,y,w,h)) for (x,y,w,h) in faces]
    areas.sort(reverse=True, key=lambda t: t[0])
    return areas[0][1]

def detect_faces_robust(gray):
    """More tolerant detection with a second pass."""
    # Equalize to help low-contrast grayscale photos
    gray_eq = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24)
    )
    if len(faces) > 0:
        return faces

    # Second pass: slightly different params; also try upscaled image for tiny faces
    up = cv2.resize(gray_eq, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_LINEAR)
    faces_up = FACE_CASCADE.detectMultiScale(
        up, scaleFactor=1.07, minNeighbors=2, minSize=(24, 24)
    )
    if len(faces_up) > 0:
        # Map boxes back to original coords
        mapped = []
        for (x, y, w, h) in faces_up:
            mapped.append((int(x/1.25), int(y/1.25), int(w/1.25), int(h/1.25)))
        return np.array(mapped, dtype=np.int32)

    return np.array([])

def fallback_center_crop(gray):
    """As a last resort, take a centered square crop so we still return a prediction."""
    h, w = gray.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    box = (x0, y0, side, side)
    return preprocess_face(gray, box)

def predict_from_gray_and_faces(gray):
    faces = detect_faces_robust(gray)
    if len(faces) == 0:
        # Last resort: centered crop
        inp = fallback_center_crop(gray)
        if inp is None:
            return None, None, None
        t0 = time.time()
        preds = MODEL.predict(inp, verbose=0)[0]
        latency_ms = int((time.time() - t0) * 1000)
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        return idx, conf, None  # no box
    else:
        box = pick_largest_face(faces)
        inp = preprocess_face(gray, box)
        if inp is None:
            return None, None, None
        t0 = time.time()
        preds = MODEL.predict(inp, verbose=0)[0]
        latency_ms = int((time.time() - t0) * 1000)
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        return idx, conf, tuple(map(int, box))

# --------------------- Routes ---------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'ok': False, 'error': 'Empty filename'}), 400

    img = Image.open(file.stream).convert('RGB')
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    idx, conf, box = predict_from_gray_and_faces(gray)
    if idx is None:
        return jsonify({'ok': True, 'status': 'no_face'})

    resp = {
        'ok': True,
        'status': 'ok',
        'label': LABELS[idx],
        'confidence': round(conf, 4),
        'latency_ms': 0
    }
    if box:
        x, y, w, h = box
        resp['box'] = [x, y, w, h]
    return jsonify(resp)

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'ok': False, 'error': 'Missing image'}), 400

    frame = decode_data_url_image(data['image'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    t0 = time.time()
    idx, conf, box = predict_from_gray_and_faces(gray)
    latency_ms = int((time.time() - t0) * 1000)

    if idx is None:
        return jsonify({'ok': True, 'status': 'no_face'})

    resp = {
        'ok': True,
        'status': 'ok',
        'label': LABELS[idx],
        'confidence': round(conf, 4),
        'latency_ms': latency_ms
    }
    if box:
        x, y, w, h = box
        resp['box'] = [x, y, w, h]
    return jsonify(resp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
