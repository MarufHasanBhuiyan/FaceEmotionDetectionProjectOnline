// ----- Upload flow -----
const upInput = document.getElementById('uploadInput');
const upBtn = document.getElementById('uploadBtn');
const upRes = document.getElementById('uploadResult');
const preview = document.getElementById('preview');

upBtn.onclick = async () => {
  const f = upInput.files?.[0];
  if (!f) { upRes.textContent = 'Pick an image first.'; return; }
  preview.src = URL.createObjectURL(f);
  const fd = new FormData();
  fd.append('image', f);
  upRes.textContent = 'Predicting...';
  const r = await fetch('/predict', { method: 'POST', body: fd });
  const j = await r.json();
  if (j.error) { upRes.textContent = 'Error: ' + j.error; return; }
  upRes.textContent = `Prediction: ${j.label} (${(j.confidence*100).toFixed(1)}%)`;
};

// ----- Webcam flow -----
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn  = document.getElementById('stopBtn');
const liveRes  = document.getElementById('liveResult');

let stream = null;
let running = false;
let sending = false;

startBtn.onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 }});
    video.srcObject = stream;
    running = true;
    liveRes.textContent = 'Running...';
    loopSend();
  } catch (e) {
    liveRes.textContent = 'Camera error: ' + e.message;
  }
};

stopBtn.onclick = () => {
  running = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  liveRes.textContent = 'Stopped.';
};

async function loopSend() {
  if (!running) return;
  if (sending) { requestAnimationFrame(loopSend); return; } // throttle

  // draw current video frame to canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

  sending = true;
  try {
    const r = await fetch('/predict-frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: dataUrl })
    });
    const j = await r.json();
    if (!j.error) {
      liveRes.textContent = `Prediction: ${j.label} (${(j.confidence*100).toFixed(1)}%)`;
    }
  } catch (e) {
    liveRes.textContent = 'Network error';
  } finally {
    sending = false;
    // send roughly ~6–10 fps to keep server light
    setTimeout(() => requestAnimationFrame(loopSend), 120);
  }
}
