// ------- small helpers -------
const $ = (sel) => document.querySelector(sel);

function makeBars(container, results) {
  container.innerHTML = "";
  const maxLen = Math.max(...results.map(r => r.prob));
  results.forEach(r => {
    const row = document.createElement("div");
    const label = document.createElement("div");
    const barWrap = document.createElement("div");
    const bar = document.createElement("div");
    const pct = (r.prob * 100).toFixed(1);

    label.className = "text-sm font-medium";
    label.textContent = `${r.label} — ${pct}%`;

    barWrap.className = "w-full h-2 bg-slate-200 rounded-full overflow-hidden";
    bar.className = "h-2 rounded-full bg-slate-900";
    bar.style.width = `${(r.prob / maxLen) * 100}%`;

    barWrap.appendChild(bar);
    row.appendChild(label);
    row.appendChild(barWrap);
    container.appendChild(row);
  });
}

async function postImageData(url, dataUrl) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function postUploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/predict_upload", { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ------- upload flow -------
(() => {
  const form = $("#uploadForm");
  const input = $("#fileInput");
  const preview = $("#preview");
  const topEl = $("#uploadTop");
  const barsEl = $("#uploadBars");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = input.files?.[0];
    if (!file) return;

    preview.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");

    topEl.textContent = "Predicting…";
    barsEl.innerHTML = "";

    try {
      const result = await postUploadFile(file);
      topEl.textContent = `Top: ${result.top.label} (${(result.top.prob * 100).toFixed(1)}%)`;
      makeBars(barsEl, result.results);
    } catch (err) {
      topEl.textContent = "Error predicting.";
      console.error(err);
    }
  });
})();

// ------- webcam flow -------
(() => {
  const video = $("#webcam");
  const canvas = $("#snapshot");
  const ctx = canvas.getContext("2d");
  const statusEl = $("#webcamStatus");
  const topEl = $("#camTop");
  const barsEl = $("#camBars");
  const startBtn = $("#startBtn");
  const stopBtn = $("#stopBtn");
  const camSelect = $("#cameraSelect");
  const flipBtn = $("#flipBtn");

  let stream = null;
  let running = false;
  let facing = "user"; // or "environment"
  let lastSent = 0;
  const throttle = (window.APP_CONFIG?.throttleMs ?? 150);

  async function listCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const vids = devices.filter(d => d.kind === "videoinput");
      camSelect.innerHTML = `<option value="">Default camera</option>`;
      vids.forEach(d => {
        const opt = document.createElement("option");
        opt.value = d.deviceId;
        opt.textContent = d.label || `Camera ${camSelect.length}`;
        camSelect.appendChild(opt);
      });
    } catch (err) {
      console.warn("enumerateDevices failed (needs permission first)", err);
    }
  }

  async function startCamera() {
    try {
      stopCamera();
      const deviceId = camSelect.value || undefined;
      const constraints = deviceId
        ? { video: { deviceId: { exact: deviceId } } }
        : { video: { facingMode: facing } };

      stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      await video.play();

      // After permission granted we can populate labels reliably
      await listCameras();

      running = true;
      statusEl.textContent = "Camera started.";
      requestAnimationFrame(loop);
    } catch (err) {
      statusEl.textContent = "Camera error (check permissions or use HTTPS).";
      console.error(err);
    }
  }

  function stopCamera() {
    running = false;
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    statusEl.textContent = "Camera stopped.";
  }

  async function loop() {
    if (!running) return;

    const now = performance.now();
    if (now - lastSent >= throttle) {
      lastSent = now;

      // draw video → canvas → toDataURL
      const w = video.videoWidth || 640;
      const h = video.videoHeight || 480;
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(video, 0, 0, w, h);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.75);

      try {
        const result = await postImageData("/predict_frame", dataUrl);
        topEl.textContent = `Top: ${result.top.label} (${(result.top.prob * 100).toFixed(1)}%)`;
        makeBars(barsEl, result.results);
      } catch (err) {
        console.error(err);
      }
    }

    requestAnimationFrame(loop);
  }

  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", stopCamera);

  camSelect.addEventListener("change", () => {
    if (running) startCamera();
  });

  flipBtn.addEventListener("click", () => {
    facing = (facing === "user") ? "environment" : "user";
    if (running && !camSelect.value) startCamera(); // only applies when not using explicit deviceId
  });

  // Helpful: ask for permission early so enumerateDevices works with labels
  if (location.protocol === "https:" || location.hostname === "localhost") {
    navigator.mediaDevices?.getUserMedia?.({ video: true })
      .then(s => {
        s.getTracks().forEach(t => t.stop());
        return listCameras();
      })
      .catch(() => {/* ignore */});
  }
})();
