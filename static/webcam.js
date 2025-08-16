const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const resultBox = document.getElementById('liveResult');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const cameraSelect = document.getElementById('cameraSelect');
const throttleChk = document.getElementById('throttleChk');

let stream = null;
let sending = false;
let rafId = null;
let lastSent = 0;
let targetIntervalMs = 200; // ~5 FPS

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
    console.error("Error listing cameras:", e);
  }
}

// Start camera with proper facingMode for mobile
async function startCamera(deviceId) {
  stopCamera();
  const constraints = {
    audio: false,
    video: deviceId 
      ? { deviceId: { exact: deviceId } } 
      : { facingMode: { ideal: "user" } } // default front
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();

  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  startSending();
}

function stopCamera() {
  stopSending();
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawBox(x, y, w, h, label, conf) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);

  const text = `${label} (${Math.round(conf * 100)}%)`;
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(x, Math.max(y - 24, 0), ctx.measureText(text).width + 10, 22);
  ctx.fillStyle = '#FFFFFF';
  ctx.font = '16px sans-serif';
  ctx.fillText(text, x + 5, Math.max(y - 8, 16));
}

async function sendFrameLoop() {
  if (!stream || !video.videoWidth) {
    rafId = requestAnimationFrame(sendFrameLoop);
    return;
  }

  const now = performance.now();
  const throttle = throttleChk.checked;
  if (sending || (throttle && now - lastSent < targetIntervalMs)) {
    rafId = requestAnimationFrame(sendFrameLoop);
    return;
  }

  sending = true;
  lastSent = now;

  const tmp = document.createElement('canvas');
  tmp.width = video.videoWidth;
  tmp.height = video.videoHeight;
  tmp.getContext('2d').drawImage(video, 0, 0, tmp.width, tmp.height);
  const dataUrl = tmp.toDataURL('image/jpeg', 0.6);

  try {
    const res = await fetch('/predict_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    });
    const out = await res.json();

    if (!out.ok) {
      resultBox.textContent = `Server error: ${out.error || 'Unknown'}`;
    } else if (out.status === 'no_face') {
      resultBox.textContent = `No face detected`;
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    } else {
      const { label, confidence, latency_ms, box } = out;
      resultBox.textContent = `Prediction: ${label} • Confidence: ${Math.round(confidence * 100)}% • Latency: ${latency_ms} ms`;
      if (box) {
        const [x, y, w, h] = box;
        drawBox(x, y, w, h, label, confidence);
      } else {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
    }
  } catch (e) {
    resultBox.textContent = `Network error`;
  } finally {
    sending = false;
    rafId = requestAnimationFrame(sendFrameLoop);
  }
}

function startSending() { if (!rafId) rafId = requestAnimationFrame(sendFrameLoop); }
function stopSending() { if (rafId) cancelAnimationFrame(rafId); rafId = null; sending = false; }

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopSending(); else startSending();
});

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

// Upload form
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
    uploadResult.textContent = `Prediction: ${out.label} • Confidence: ${Math.round(out.confidence*100)}% • Latency: ${out.latency_ms} ms`;
  } catch (err) {
    uploadResult.textContent = 'Network error';
  }
});

// Populate camera list at load
navigator.mediaDevices?.getUserMedia({ video: true, audio: false })
  .then(() => listCameras())
  .catch(() => listCameras());
