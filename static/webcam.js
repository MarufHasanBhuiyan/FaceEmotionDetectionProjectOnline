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
let populatedOnce = false;

// ---------- Cameras ----------
async function primePermissions() {
  // Ask once so labels become available (esp. iOS).
  try {
    const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    tmp.getTracks().forEach(t => t.stop());
  } catch (_) { /* ignore */ }
}

async function listCameras(preserveSelection = true) {
  try {
    const prev = cameraSelect.value;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d => d.kind === 'videoinput');

    cameraSelect.innerHTML = '';
    cams.forEach((d, i) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId || '';
      const label = d.label || `Camera ${i + 1}`;
      opt.text = label;
      // Useful hint for facing mode:
      opt.dataset.facing =
        /back|rear|environment/i.test(label) ? 'environment' :
        /front|user|face/i.test(label) ? 'user' : '';
      cameraSelect.appendChild(opt);
    });

    if (preserveSelection && [...cameraSelect.options].some(o => o.value === prev)) {
      cameraSelect.value = prev;
    } else {
      // Prefer a rear camera by default on phones
      const rear = [...cameraSelect.options].find(o => o.dataset.facing === 'environment');
      if (rear) cameraSelect.value = rear.value;
    }
    populatedOnce = true;
  } catch (e) {
    console.error('Error listing cameras:', e);
  }
}

async function startCamera(deviceId) {
  stopCamera();

  const facingHint =
    cameraSelect.selectedOptions[0]?.dataset?.facing || undefined;

  // Build constraints: prefer exact deviceId, but also provide facingMode hint.
  const constraints = {
    audio: false,
    video: deviceId
      ? { deviceId: { exact: deviceId }, facingMode: facingHint ? { ideal: facingHint } : undefined }
      : { facingMode: { ideal: facingHint || 'user' } }
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

// Refresh camera list if devices change (USB cam plug/unplug, mobile rotate, etc.)
navigator.mediaDevices?.addEventListener?.('devicechange', () => {
  listCameras(true);
});

// ---------- Drawing ----------
function drawBox(x, y, w, h, label, conf) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);

  const text = `${label} (${Math.round(conf * 100)}%)`;
  ctx.font = '16px sans-serif';
  const tw = ctx.measureText(text).width + 10;

  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(x, Math.max(y - 24, 0), tw, 22);
  ctx.fillStyle = '#FFFFFF';
  ctx.fillText(text, x + 5, Math.max(y - 8, 16));
}

// ---------- Loop ----------
async function sendFrameLoop() {
  if (!stream || !video.videoWidth) {
    rafId = requestAnimationFrame(sendFrameLoop);
    return;
  }

  const now = performance.now();
  if (sending || (throttleChk.checked && now - lastSent < targetIntervalMs)) {
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
      resultBox.textContent = 'No face detected';
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    } else {
      const { label, confidence, latency_ms, box } = out;
      resultBox.textContent =
        `Prediction: ${label} • Confidence: ${Math.round(confidence * 100)}% • Latency: ${latency_ms} ms`;
      if (box) {
        const [x, y, w, h] = box;
        drawBox(x, y, w, h, label, confidence);
      } else {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
    }
  } catch (_) {
    resultBox.textContent = 'Network error';
  } finally {
    sending = false;
    rafId = requestAnimationFrame(sendFrameLoop);
  }
}

function startSending() {
  if (!rafId) rafId = requestAnimationFrame(sendFrameLoop);
}
function stopSending() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  sending = false;
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopSending(); else startSending();
});

// ---------- UI ----------
startBtn.addEventListener('click', async () => {
  try {
    // Don’t repopulate here; that was overriding your selection
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
    uploadResult.textContent =
      `Prediction: ${out.label} • Confidence: ${Math.round(out.confidence * 100)}% • Latency: ${out.latency_ms} ms`;
  } catch (_) {
    uploadResult.textContent = 'Network error';
  }
});

// Initial setup: ask permission once, then populate the list
(async () => {
  await primePermissions();
  await listCameras(false);
})();
