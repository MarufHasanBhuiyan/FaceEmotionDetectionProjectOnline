// ----- DOM -----
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const resultBox = document.getElementById('liveResult');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const cameraSelect = document.getElementById('cameraSelect');
const throttleChk = document.getElementById('throttleChk');

// ----- State / tuning -----
let stream = null;
let sending = false;
let rafId = null;
let lastSent = 0;

// ~5 FPS when throttling is on
const TARGET_INTERVAL_MS = 200;

// Send smaller frames to the server to cut latency/bandwidth
const CAPTURE_WIDTH = 320;
const CAPTURE_HEIGHT = 240;

// Ignore very tiny face boxes (likely false positives from Haar)
const MIN_BOX_AREA_RATIO = 0.01; // 1% of frame

// Offscreen canvas reused for capture (no re-alloc each frame)
const tmp = document.createElement('canvas');
tmp.width = CAPTURE_WIDTH;
tmp.height = CAPTURE_HEIGHT;
const tctx = tmp.getContext('2d', { willReadFrequently: true });

// ----- Utils -----
async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d => d.kind === 'videoinput');
    cameraSelect.innerHTML = '';
    cams.forEach((d, i) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.text = d.label || `Camera ${i + 1}`;
      cameraSelect.appendChild(opt);
    });
  } catch (e) {
    console.error('Error listing cameras:', e);
  }
}

function stopSending() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  sending = false;
}

function stopCamera() {
  stopSending();
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function setOverlayToVideo() {
  // Size overlay to match the *displayed* video
  overlay.width = video.videoWidth || CAPTURE_WIDTH;
  overlay.height = video.videoHeight || CAPTURE_HEIGHT;
}

async function startCamera(deviceId) {
  stopCamera();

  const constraints = {
    audio: false,
    video: deviceId
      ? {
          deviceId: { exact: deviceId },
          width: { ideal: 640, max: 640 },
          height: { ideal: 480, max: 480 }
        }
      : {
          facingMode: { ideal: 'user' }, // default to front if none selected
          width: { ideal: 640, max: 640 },
          height: { ideal: 480, max: 480 }
        }
  };

  // Open the camera
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;

  // Wait for actual dimensions
  await video.play();
  if (!video.videoWidth) {
    await new Promise(res => video.addEventListener('loadedmetadata', res, { once: true }));
  }

  setOverlayToVideo();
  startSending();
}

function startSending() {
  if (!rafId) rafId = requestAnimationFrame(sendFrameLoop);
}

function drawBoxScaled(box, label, conf) {
  const [x, y, w, h] = box;

  // The server saw CAPTURE_WIDTH x CAPTURE_HEIGHT.
  // Scale to overlay/video size for correct drawing.
  const sx = overlay.width / CAPTURE_WIDTH;
  const sy = overlay.height / CAPTURE_HEIGHT;
  const X = Math.round(x * sx);
  const Y = Math.round(y * sy);
  const W = Math.round(w * sx);
  const H = Math.round(h * sy);

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 2;
  ctx.strokeRect(X, Y, W, H);

  const text = `${label} (${Math.round(conf * 100)}%)`;
  ctx.font = '16px sans-serif';
  const tw = ctx.measureText(text).width + 10;
  const th = 22;

  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(X, Math.max(Y - th, 0), tw, th);
  ctx.fillStyle = '#FFFFFF';
  ctx.fillText(text, X + 5, Math.max(Y - 6, 16));
}

async function sendFrameLoop() {
  if (!stream || !video.videoWidth) {
    rafId = requestAnimationFrame(sendFrameLoop);
    return;
  }

  const now = performance.now();
  const throttle = throttleChk.checked;
  if (sending || (throttle && now - lastSent < TARGET_INTERVAL_MS)) {
    rafId = requestAnimationFrame(sendFrameLoop);
    return;
  }
  sending = true;
  lastSent = now;

  // Draw the current frame to a *small* canvas to reduce payload
  tctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
  const dataUrl = tmp.toDataURL('image/jpeg', 0.5); // compress further to reduce latency

  try {
    const res = await fetch('/predict_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    });
    const out = await res.json();

    if (!out.ok) {
      resultBox.textContent = `Server error: ${out.error || 'Unknown'}`;
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    } else if (out.status === 'no_face') {
      resultBox.textContent = 'No face detected';
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    } else {
      const { label, confidence, latency_ms, box } = out;

      // Extra client-side guard: ignore tiny boxes (likely FP)
      if (box) {
        const [bx, by, bw, bh] = box;
        const boxArea = bw * bh;
        const frameArea = CAPTURE_WIDTH * CAPTURE_HEIGHT;
        const ratio = boxArea / frameArea;
        if (ratio < MIN_BOX_AREA_RATIO) {
          resultBox.textContent = 'No face detected';
          ctx.clearRect(0, 0, overlay.width, overlay.height);
        } else {
          resultBox.textContent = `Prediction: ${label} • Confidence: ${Math.round(confidence * 100)}% • Latency: ${latency_ms} ms`;
          drawBoxScaled(box, label, confidence);
        }
      } else {
        resultBox.textContent = `Prediction: ${label} • Confidence: ${Math.round(confidence * 100)}% • Latency: ${latency_ms} ms`;
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
    }
  } catch (e) {
    resultBox.textContent = 'Network error';
    // Keep going; network can be transient
  } finally {
    sending = false;
    rafId = requestAnimationFrame(sendFrameLoop);
  }
}

// Pause network work when tab not visible
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopSending();
  } else if (stream) {
    startSending();
  }
});

// ----- UI wiring -----
startBtn.addEventListener('click', async () => {
  try {
    await listCameras();
    const selectedDeviceId = cameraSelect.value || undefined;
    await startCamera(selectedDeviceId);
    startBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (e) {
    alert('Camera error: ' + e.message);
  }
});

stopBtn.addEventListener('click', () => {
  stopCamera();
  startBtn.disabled = false;
  stopBtn.disabled = true;
});

// Upload form (unchanged)
const uploadForm = document.getElementById('uploadForm');
const uploadResult = document.getElementById('uploadResult');
const uploadedPreview = document.getElementById('uploadedPreview');

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const data = new FormData(uploadForm);
  uploadResult.textContent = 'Predicting...';
  uploadedPreview.style.display = 'none';

  const file = uploadForm.querySelector('input[type=file]').files[0];
  if (file) {
    uploadedPreview.src = URL.createObjectURL(file);
    uploadedPreview.style.display = 'block';
  }

  try {
    const res = await fetch('/predict_image', { method: 'POST', body: data });
    const out = await res.json();

    if (!out.ok) {
      uploadResult.textContent = `Server error: ${out.error || 'Unknown'}`;
      return;
    }
    if (out.status === 'no_face') {
      uploadResult.textContent = 'No face detected';
      return;
    }
    uploadResult.textContent = `Prediction: ${out.label} • Confidence: ${Math.round(out.confidence * 100)}% • Latency: ${out.latency_ms} ms`;
  } catch (err) {
    uploadResult.textContent = 'Network error';
  }
});

// Ask permission once so labels populate on mobile, then list cams
navigator.mediaDevices?.getUserMedia({ video: true, audio: false })
  .then(() => listCameras())
  .catch(() => listCameras());
