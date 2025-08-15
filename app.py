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
# Labels in the exact order your model outputs (8 classes)
LABELS = ['angry','depression','disgust','fear','happy','neutral','sad','surprise']

# Haarcascade for face detection
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load model once at startup
# If your model was saved with model.save('emotiondetector.h5')
# this will restore the full model (architecture + weights).
MODEL = load_model('emotiondetector.h5')

# Warmup a dummy call (optional, improves first request latency)
_ = MODEL.predict(np.zeros((1,48,48,1), dtype=np.float32))

app = Flask(__name__)

# --------------------- Helpers ---------------------
def preprocess_face(gray_image, box):
    x,y,w,h = box
    face = gray_image[y:y+h, x:x+w]
    # Guard against empty slices
    if face.size == 0:
        return None
    face = cv2.resize(face, (48,48), interpolation=cv2.INTER_AREA)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(img_to_array(face), axis=0)  # (1,48,48,1)
    return face

def decode_data_url_image(data_url):
    # data_url: "data:image/jpeg;base64,......"
    header, b64data = data_url.split(',', 1)
    img_bytes = base64.b64decode(b64data)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # Convert to OpenCV BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def pick_largest_face(faces):
    if len(faces) == 0:
        return None
    # faces: array of (x,y,w,h). Pick by area
    areas = [(w*h, (x,y,w,h)) for (x,y,w,h) in faces]
    areas.sort(reverse=True, key=lambda t: t[0])
    return areas[0][1]

# --------------------- Routes ---------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """Handles file upload (image)"""
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'No file'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'ok': False, 'error': 'Empty filename'}), 400

    # Read image
    img = Image.open(file.stream).convert('RGB')
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Detect faces on grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'ok': True, 'status': 'no_face'})

    box = pick_largest_face(faces)
    inp = preprocess_face(gray, box)
    if inp is None:
        return jsonify({'ok': True, 'status': 'no_face'})

    t0 = time.time()
    preds = MODEL.predict(inp, verbose=0)[0]
    latency_ms = int((time.time() - t0)*1000)
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    x,y,w,h = map(int, box)
    return jsonify({
        'ok': True,
        'status': 'ok',
        'label': LABELS[idx],
        'confidence': round(conf, 4),
        'latency_ms': latency_ms,
        'box': [x,y,w,h]
    })

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Handles base64-encoded frame from webcam"""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'ok': False, 'error': 'Missing image'}), 400

    frame = decode_data_url_image(data['image'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'ok': True, 'status': 'no_face'})

    box = pick_largest_face(faces)
    inp = preprocess_face(gray, box)
    if inp is None:
        return jsonify({'ok': True, 'status': 'no_face'})

    t0 = time.time()
    preds = MODEL.predict(inp, verbose=0)[0]
    latency_ms = int((time.time() - t0)*1000)
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    x,y,w,h = map(int, box)

    return jsonify({
        'ok': True,
        'status': 'ok',
        'label': LABELS[idx],
        'confidence': round(conf, 4),
        'latency_ms': latency_ms,
        'box': [x,y,w,h]
    })

if __name__ == '__main__':
    # Local run (Render will use Gunicorn)
    app.run(host='0.0.0.0', port=5000)
