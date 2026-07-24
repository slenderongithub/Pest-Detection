/* LeafScan — fully in-browser inference (onnxruntime-web).
 *
 * Same UI + result shape as the Starlette SPA, but there is NO backend: the model runs in
 * WASM in the browser. The serving contract (classes, normalization, temperature, abstain
 * threshold, knowledge) is read from model/meta.json — never hardcoded — so this matches the
 * Python pipeline. Grad-CAM (needs autograd) is replaced by gradient-free CAM using the fc
 * weights, which is equivalent for a global-avg-pool ResNet.
 */
(() => {
  "use strict";
  const RING_CIRC = 2 * Math.PI * 52;
  const FRAME_MS = 1200;
  const MODEL_DIR = "model/";
  const CAM_MAX_SIDE = 768; // ponytail: cap overlay/box math resolution; raise if you want sharper boxes

  const $ = (id) => document.getElementById(id);
  const el = {
    root: document.documentElement,
    themeToggle: $("theme-toggle"), brandChip: $("brand-chip"),
    demoBanner: $("demo-banner"), demoMsg: $("demo-msg"),
    statArch: $("stat-arch"), statClasses: $("stat-classes"), statF1: $("stat-f1"),
    camStage: $("cam-stage"), camFeed: $("camera-feed"), canvas: $("capture-canvas"),
    camStatus: $("cam-status"), camRec: $("cam-rec"), scanline: $("scanline"), camIdle: $("cam-idle"),
    startCam: $("start-camera"), capture: $("capture"), stopCam: $("stop-camera"),
    drop: $("drop"), fileInput: $("file-input"), dropTitle: $("drop-title"),
    resultChip: $("result-chip"), resultEmpty: $("result-empty"), resultBody: $("result-body"),
    ringFill: $("ring-fill"), confNum: $("conf-num"), resultName: $("result-name"),
    resultSub: $("result-sub"), sevChip: $("severity-chip"), sevText: $("severity-text"), calibText: $("calib-text"),
    focusStage: $("focus-stage"), focusImg: $("focus-img"), focusPlaceholder: $("focus-placeholder"), focusRegions: $("focus-regions"),
    top3: $("top3"),
    pestEmpty: $("pest-empty"), pestBody: $("pest-body"), pestCommon: $("pest-common"), pestType: $("pest-type"), pestDesc: $("pest-desc"), pestIpm: $("pest-ipm"),
    treatmentEmpty: $("treatment-empty"), treatmentBody: $("treatment-body"), treatmentRows: $("treatment-rows"),
    history: $("history"),
    uncertain: $("uncertain"), uncDetail: $("unc-detail"),
    disclaimer: $("disclaimer"),
  };

  const state = { stream: null, timer: null, rec: null, busy: false, demo: false, history: [], recStart: 0 };
  const engine = { session: null, meta: null, fc: null, ready: false };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const sevVar = (sev) => `var(--sev-${String(sev || "med").toLowerCase().replace("critical", "crit")})`;

  /* ---------- theme ---------- */
  function applyTheme(t) { el.root.setAttribute("data-theme", t); }
  (function initTheme() {
    const saved = localStorage.getItem("leafscan-theme");
    const sys = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    applyTheme(saved || sys);
  })();
  el.themeToggle.addEventListener("click", () => {
    const next = el.root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next); localStorage.setItem("leafscan-theme", next);
  });

  /* ---------- demo / no-model ---------- */
  function showDemo(message) {
    state.demo = true;
    el.demoBanner.hidden = false;
    if (message) el.demoMsg.textContent = message;
    el.brandChip.innerHTML = '<span class="dot" style="background:var(--amber)"></span> demo mode';
    el.startCam.disabled = true; el.capture.disabled = true;
    el.dropTitle.textContent = "Model unavailable";
    el.drop.style.pointerEvents = "none"; el.drop.style.opacity = ".55";
  }

  /* ---------- rendering (identical to the server SPA) ---------- */
  function setConfidence(pct) {
    el.ringFill.style.strokeDashoffset = String(RING_CIRC * (1 - Math.max(0, Math.min(100, pct)) / 100));
    let cur = 0;
    const step = () => {
      cur += (pct - cur) * 0.12;
      if (pct - cur < 0.2) cur = pct;
      el.confNum.textContent = cur.toFixed(0) + "%";
      if (cur < pct) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  function renderTop3(items = []) {
    if (!items.length) { el.top3.innerHTML = '<p class="muted-note">No scan yet.</p>'; return; }
    el.top3.innerHTML = items.map((it, i) => {
      const w = Math.max(0, Math.min(100, Number(it.probability ?? 0)));
      const nm = esc(it.display_label ?? it.label ?? "Unknown");
      const dim = i === 0 ? "" : "dim";
      return `<div class="bar-row"><div class="bar-top"><span class="nm ${dim}">${nm}</span>
        <span class="pct">${w.toFixed(1)}%</span></div>
        <div class="bar-track"><div class="bar-fill ${dim}" data-w="${w}"></div></div></div>`;
    }).join("");
    requestAnimationFrame(() => el.top3.querySelectorAll(".bar-fill").forEach((b) => { b.style.width = b.dataset.w + "%"; }));
  }
  function renderPest(info) {
    if (!info) { el.pestBody.hidden = true; el.pestEmpty.hidden = false; return; }
    el.pestEmpty.hidden = true; el.pestBody.hidden = false;
    el.pestCommon.textContent = info.common_name || "—";
    el.pestType.textContent = info.pest_type || "—";
    el.pestDesc.textContent = info.description || "";
    if (info.ipm) { el.pestIpm.hidden = false; el.pestIpm.innerHTML = `<b>IPM:</b> ${esc(info.ipm)}`; }
    else el.pestIpm.hidden = true;
  }
  function renderTreatment(info) {
    const rows = info && info.pesticides ? info.pesticides : [];
    if (!rows.length) { el.treatmentBody.hidden = true; el.treatmentEmpty.hidden = false; return; }
    el.treatmentEmpty.hidden = true; el.treatmentBody.hidden = false;
    el.treatmentRows.innerHTML = rows.map((p) =>
      `<tr><td><span class="pn">${esc(p.name)}</span></td><td><span class="mono">${esc(p.dose)}</span></td></tr>`).join("");
  }
  function pushHistory(r) {
    state.history.unshift({ label: r.display_label || r.label || "Unknown", pct: Number(r.confidence ?? 0), sev: r.severity, abstained: r.abstained });
    state.history = state.history.slice(0, 5);
    el.history.innerHTML = state.history.map((h) => {
      const c = sevVar(h.sev);
      const tag = h.abstained ? '<span class="tm">abstained</span>' : `<span class="tm">${h.pct.toFixed(0)}% confident</span>`;
      return `<div class="hist-item">
        <div class="hist-thumb" style="background:linear-gradient(135deg, ${c}, color-mix(in srgb, ${c} 45%, #000))"></div>
        <div class="hist-meta"><div class="nm">${esc(h.label)}</div>${tag}</div>
        <span class="hist-pct">${h.pct.toFixed(0)}%</span>
        <span class="hist-sev" style="background:${c}"></span></div>`;
    }).join("");
  }
  function renderFocus(result) {
    if (result.overlay_data_url) { el.focusImg.src = result.overlay_data_url; el.focusImg.hidden = false; el.focusPlaceholder.hidden = true; }
    const n = (result.boxes || []).length;
    el.focusRegions.hidden = false;
    el.focusRegions.textContent = n === 1 ? "1 region" : `${n} regions`;
  }
  function renderResult(result, withFocus) {
    el.resultEmpty.hidden = true; el.resultBody.hidden = false; el.resultChip.hidden = false;
    const conf = Number(result.confidence ?? 0);
    setConfidence(conf);
    el.resultName.textContent = result.display_label || result.label || "Unknown";
    el.resultSub.textContent = result.pest_info ? (result.pest_info.pest_type || "") : "";
    const sev = result.severity || "Medium";
    el.sevChip.style.setProperty("--sev-color", sevVar(sev));
    el.sevText.textContent = result.severity ? `Severity · ${sev}` : "Severity · n/a";
    const thr = Number(result.abstain_threshold ?? 40);
    const temp = result.temperature != null ? ` (T=${Number(result.temperature).toFixed(2)})` : "";
    if (result.abstained) {
      el.resultChip.innerHTML = '<span class="dot" style="background:var(--amber)"></span> uncertain';
      el.calibText.textContent = `Below the ${thr.toFixed(0)}% decision threshold — treat as a weak hint.`;
      el.uncertain.hidden = false;
      el.uncDetail.textContent = `${conf.toFixed(1)}% < ${thr.toFixed(0)}% threshold`;
    } else {
      el.resultChip.innerHTML = '<span class="dot"></span> confident';
      el.calibText.textContent = `Temperature-calibrated${temp} · above the ${thr.toFixed(0)}% decision threshold.`;
      el.uncertain.hidden = true;
    }
    renderPest(result.pest_info); renderTreatment(result.pest_info); renderTop3(result.top_predictions);
    if (withFocus) renderFocus(result);
    if (result.disclaimer) el.disclaimer.textContent = result.disclaimer;
    pushHistory(result);
  }

  /* ---------- inference engine (replaces the network layer) ---------- */
  const prettyLabel = (s) => String(s).replace(/_/g, " ").trim();

  function pestInfo(label) {
    const pests = engine.meta.pests || {};
    let e = pests[label];
    if (!e) { const key = Object.keys(pests).find((k) => k.toLowerCase() === label.toLowerCase()); e = key ? pests[key] : null; }
    if (!e) return null;
    return {
      common_name: e.common_name || label, pest_type: e.pest_type || "Unknown",
      severity: e.severity || "Medium", description: (e.description || "").trim(),
      pesticides: (e.pesticides || []).map((p) => ({ name: p.name || "", dose: p.dose || "" })),
      ipm: (e.ipm || "").trim(),
    };
  }

  // Preprocess to match leafscan/preprocess.eval_transform: Resize(shorter=input*256/224) →
  // CenterCrop(input) → ToTensor → Normalize. ponytail: canvas bilinear ≈ torchvision's
  // antialiased resize (not bit-exact); top-1 is stable but expect tiny prob drift vs Python.
  async function preprocess(bitmap) {
    const { input_size, resize_ratio, mean, std } = engine.meta;
    const short = Math.round(input_size * resize_ratio);
    const scale = short / Math.min(bitmap.width, bitmap.height);
    const rw = Math.max(input_size, Math.round(bitmap.width * scale));
    const rh = Math.max(input_size, Math.round(bitmap.height * scale));
    const c = document.createElement("canvas"); c.width = rw; c.height = rh;
    const ctx = c.getContext("2d"); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, 0, 0, rw, rh);
    const left = Math.max(0, (rw - input_size) >> 1), top = Math.max(0, (rh - input_size) >> 1);
    const { data } = ctx.getImageData(left, top, input_size, input_size);
    const n = input_size * input_size;
    const t = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      t[i] = (data[i * 4] / 255 - mean[0]) / std[0];
      t[n + i] = (data[i * 4 + 1] / 255 - mean[1]) / std[1];
      t[2 * n + i] = (data[i * 4 + 2] / 255 - mean[2]) / std[2];
    }
    return new ort.Tensor("float32", t, [1, 3, input_size, input_size]);
  }

  function softmaxT(logits, T) {
    const s = logits.map((x) => x / Math.max(T, 1e-6));
    const m = Math.max(...s);
    const ex = s.map((x) => Math.exp(x - m));
    const sum = ex.reduce((a, b) => a + b, 0);
    return ex.map((x) => x / sum);
  }

  // gradient-free CAM: cam[h,w] = relu(sum_k fc[c,k]*feat[k,h,w]), normalized to [0,1].
  function computeCam(feat, dims, classIndex) {
    const [, C, H, W] = dims, hw = H * W, base = classIndex * C;
    const cam = new Float32Array(hw);
    for (let k = 0; k < C; k++) { const w = engine.fc[base + k], off = k * hw; for (let i = 0; i < hw; i++) cam[i] += w * feat[off + i]; }
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < hw; i++) { if (cam[i] < 0) cam[i] = 0; if (cam[i] < mn) mn = cam[i]; if (cam[i] > mx) mx = cam[i]; }
    const rng = (mx - mn) || 1e-8;
    for (let i = 0; i < hw; i++) cam[i] = (cam[i] - mn) / rng;
    return { cam, H, W };
  }

  // Project the HxW CAM onto the original image geometry (undo resize+centercrop), at a
  // working resolution capped by CAM_MAX_SIDE. Returns {gray:Float32[0..1], ow, oh}.
  function projectCam({ cam, H, W }, imgW, imgH, inputSize, resizeRatio) {
    const long = Math.max(imgW, imgH), s = long > CAM_MAX_SIDE ? CAM_MAX_SIDE / long : 1;
    const ow = Math.max(1, Math.round(imgW * s)), oh = Math.max(1, Math.round(imgH * s));
    // 1) upsample tiny cam → inputSize (bilinear via canvas)
    const small = document.createElement("canvas"); small.width = W; small.height = H;
    const sctx = small.getContext("2d"), sd = sctx.createImageData(W, H);
    for (let i = 0; i < W * H; i++) { const v = Math.round(cam[i] * 255); sd.data[i * 4] = sd.data[i * 4 + 1] = sd.data[i * 4 + 2] = v; sd.data[i * 4 + 3] = 255; }
    sctx.putImageData(sd, 0, 0);
    const inp = document.createElement("canvas"); inp.width = inputSize; inp.height = inputSize;
    const ictx = inp.getContext("2d"); ictx.imageSmoothingEnabled = true; ictx.imageSmoothingQuality = "high";
    ictx.drawImage(small, 0, 0, inputSize, inputSize);
    // 2) place the inputSize block into the resized canvas at the centre-crop offset
    const short = Math.round(inputSize * resizeRatio), sc = short / Math.min(imgW, imgH);
    const rw = Math.max(inputSize, Math.round(imgW * sc)), rh = Math.max(inputSize, Math.round(imgH * sc));
    const left = Math.max(0, (rw - inputSize) >> 1), top = Math.max(0, (rh - inputSize) >> 1);
    const big = document.createElement("canvas"); big.width = rw; big.height = rh;
    const bctx = big.getContext("2d"); bctx.fillStyle = "#000"; bctx.fillRect(0, 0, rw, rh);
    bctx.drawImage(inp, left, top);
    // 3) scale the resized canvas down to the (capped) original size
    const out = document.createElement("canvas"); out.width = ow; out.height = oh;
    const octx = out.getContext("2d"); octx.imageSmoothingEnabled = true; octx.imageSmoothingQuality = "high";
    octx.drawImage(big, 0, 0, rw, rh, 0, 0, ow, oh);
    const px = octx.getImageData(0, 0, ow, oh).data, gray = new Float32Array(ow * oh);
    for (let i = 0; i < ow * oh; i++) gray[i] = px[i * 4] / 255;
    return { gray, ow, oh };
  }

  function quantile(arr, q) { const a = Float32Array.from(arr).sort(); return a[Math.min(a.length - 1, Math.floor(q * (a.length - 1)))]; }

  // Port of postprocess.boxes_from_cam + connected_component_boxes (BFS).
  function boxesFromCam(gray, W, H) {
    const thr = Math.max(0.5, quantile(gray, 0.85));
    const minPixels = Math.max(24, Math.floor(0.002 * W * H));
    const pad = Math.max(3, Math.floor(0.012 * Math.max(W, H)));
    const visited = new Uint8Array(W * H), comps = [];
    const qx = new Int32Array(W * H), qy = new Int32Array(W * H);
    for (let sy = 0; sy < H; sy++) for (let sx = 0; sx < W; sx++) {
      const idx = sy * W + sx;
      if (visited[idx] || gray[idx] < thr) continue;
      let head = 0, tail = 0; qx[tail] = sx; qy[tail] = sy; tail++; visited[idx] = 1;
      let n = 0, minx = sx, maxx = sx, miny = sy, maxy = sy;
      while (head < tail) {
        const x = qx[head], y = qy[head]; head++; n++;
        if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y;
        const nb = [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]];
        for (const [nx, ny] of nb) {
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          const ni = ny * W + nx;
          if (!visited[ni] && gray[ni] >= thr) { visited[ni] = 1; qx[tail] = nx; qy[tail] = ny; tail++; }
        }
      }
      if (n >= minPixels) comps.push([n, minx, miny, maxx, maxy]);
    }
    comps.sort((a, b) => b[0] - a[0]);
    let boxes = comps.slice(0, 8).map(([, x1, y1, x2, y2]) => ({
      x1: Math.max(0, x1 - pad), y1: Math.max(0, y1 - pad), x2: Math.min(W - 1, x2 + pad), y2: Math.min(H - 1, y2 + pad),
    }));
    if (!boxes.length) { // fallback: small box around the single hottest peak
      let mx = -1, mi = 0; for (let i = 0; i < gray.length; i++) if (gray[i] > mx) { mx = gray[i]; mi = i; }
      if (mx <= 0) return [];
      const px = mi % W, py = (mi / W) | 0, side = Math.max(8, Math.floor(0.12 * Math.min(W, H)));
      boxes = [{ x1: Math.max(0, px - side), y1: Math.max(0, py - side), x2: Math.min(W - 1, px + side), y2: Math.min(H - 1, py + side) }];
    }
    return boxes;
  }

  // Port of postprocess.overlay_image: tinted heatmap + rounded label boxes → JPEG data URL.
  function overlay(bitmap, gray, W, H, boxes) {
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const ctx = c.getContext("2d");
    ctx.imageSmoothingQuality = "high"; ctx.drawImage(bitmap, 0, 0, W, H);
    const img = ctx.getImageData(0, 0, W, H), d = img.data;
    for (let i = 0; i < W * H; i++) {
      const a = Math.min(185, Math.pow(gray[i], 1.45) * 185) / 255;
      d[i * 4] = d[i * 4] * (1 - a) + 255 * a;
      d[i * 4 + 1] = d[i * 4 + 1] * (1 - a) + 84 * a;
      d[i * 4 + 2] = d[i * 4 + 2] * (1 - a) + 48 * a;
    }
    ctx.putImageData(img, 0, 0);
    ctx.strokeStyle = "rgba(255,226,130,1)"; ctx.lineWidth = 5;
    for (const b of boxes) { roundRect(ctx, b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1, 12); ctx.stroke(); }
    return c.toDataURL("image/jpeg", 0.9);
  }
  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2); ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  async function infer(blob, withFocus) {
    const bitmap = await createImageBitmap(blob);
    const meta = engine.meta;
    const tensor = await preprocess(bitmap);
    const out = await engine.session.run({ input: tensor });
    const logits = Array.from(out.logits.data);
    const probs = softmaxT(logits, meta.temperature);
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const top = order[0][1];
    const label = meta.class_names[top];
    const confidence = probs[top] * 100;
    const threshold = meta.abstain_threshold * 100;
    const abstained = confidence < threshold;
    const info = pestInfo(label);

    const result = {
      model_name: meta.model_name, label, display_label: prettyLabel(label),
      confidence: +confidence.toFixed(2), severity: info ? info.severity : null,
      abstained, abstain_threshold: +threshold.toFixed(2), temperature: +Number(meta.temperature).toFixed(4),
      top_predictions: order.slice(0, 3).map(([p, i]) => ({ label: meta.class_names[i], display_label: prettyLabel(meta.class_names[i]), probability: +(p * 100).toFixed(2) })),
      boxes: [], class_names: meta.class_names, pest_info: info,
      image_size: { width: bitmap.width, height: bitmap.height }, disclaimer: meta.disclaimer, overlay_data_url: null,
    };

    // CAM overlay only on the upload/capture path and only when confident (mirrors Python gate).
    if (withFocus && !abstained) {
      const camRaw = computeCam(out.feat.data, out.feat.dims, top);
      const proj = projectCam(camRaw, bitmap.width, bitmap.height, meta.input_size, meta.resize_ratio);
      const boxes = boxesFromCam(proj.gray, proj.ow, proj.oh);
      result.boxes = boxes;
      if (boxes.length) {
        // draw the overlay at the working (capped) resolution so boxes register
        const scaled = await createImageBitmap(bitmap, { resizeWidth: proj.ow, resizeHeight: proj.oh, resizeQuality: "high" });
        result.overlay_data_url = overlay(scaled, proj.gray, proj.ow, proj.oh, boxes);
      }
    }
    return result;
  }

  async function analyze(blob, _endpoint, withFocus) {
    if (state.busy || state.demo || !engine.ready) return;
    state.busy = true;
    try { renderResult(await infer(blob, withFocus), withFocus); }
    catch (e) { console.error("inference failed", e); }
    finally { state.busy = false; }
  }

  /* ---------- upload ---------- */
  el.fileInput.addEventListener("change", () => {
    const f = el.fileInput.files && el.fileInput.files[0];
    if (f) { el.dropTitle.textContent = f.name.slice(0, 28); analyze(f, null, true); }
  });
  ["dragover", "dragenter"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => el.drop.addEventListener(ev, () => el.drop.classList.remove("drag")));
  el.drop.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) { el.dropTitle.textContent = f.name.slice(0, 28); analyze(f, null, true); }
  });

  /* ---------- camera ---------- */
  function captureFrame(withFocus) {
    if (!state.stream || !el.camFeed.videoWidth) return;
    const c = el.canvas; c.width = el.camFeed.videoWidth; c.height = el.camFeed.videoHeight;
    const ctx = c.getContext("2d");
    ctx.save(); ctx.scale(-1, 1); ctx.drawImage(el.camFeed, -c.width, 0, c.width, c.height); ctx.restore();
    c.toBlob((b) => b && analyze(b, null, withFocus), "image/jpeg", 0.9);
  }
  async function startCamera() {
    if (state.demo) return;
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false });
      el.camFeed.srcObject = state.stream; await el.camFeed.play();
      el.camStage.classList.add("on"); el.scanline.hidden = false; el.camRec.hidden = false;
      el.startCam.hidden = true; el.capture.hidden = false; el.stopCam.hidden = false;
      el.camStatus.innerHTML = '<span class="live-dot"></span> live'; state.recStart = 0;
      state.rec = setInterval(() => { state.recStart++; const m = String(Math.floor(state.recStart / 60)).padStart(2, "0"); const s = String(state.recStart % 60).padStart(2, "0"); el.camRec.innerHTML = `<span class="live-dot"></span> REC ${m}:${s}`; }, 1000);
      state.timer = setInterval(() => captureFrame(false), FRAME_MS); // frames: no CAM (latency)
      captureFrame(false);
    } catch (e) { el.camStatus.innerHTML = '<span class="dot" style="background:var(--sev-high)"></span> unavailable'; console.error("camera error", e); }
  }
  function stopCamera() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    if (state.rec) { clearInterval(state.rec); state.rec = null; }
    if (state.stream) { state.stream.getTracks().forEach((t) => t.stop()); state.stream = null; }
    el.camFeed.srcObject = null; el.camStage.classList.remove("on");
    el.scanline.hidden = true; el.camRec.hidden = true;
    el.startCam.hidden = false; el.capture.hidden = true; el.stopCam.hidden = true;
    el.camStatus.innerHTML = '<span class="dot"></span> idle';
  }
  el.startCam.addEventListener("click", startCamera);
  el.stopCam.addEventListener("click", stopCamera);
  el.capture.addEventListener("click", () => captureFrame(true)); // manual capture: with CAM
  window.addEventListener("beforeunload", stopCamera);

  /* ---------- boot: load model in the browser ---------- */
  (async function boot() {
    el.brandChip.innerHTML = '<span class="dot"></span> loading model…';
    try {
      ort.env.wasm.numThreads = 1; // single-threaded WASM → no COOP/COEP headers needed
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
      const meta = await (await fetch(MODEL_DIR + "meta.json")).json();
      engine.meta = meta;
      const [session, fcBuf] = await Promise.all([
        ort.InferenceSession.create(MODEL_DIR + meta.onnx, { executionProviders: ["wasm"] }),
        fetch(MODEL_DIR + "fc.bin").then((r) => r.arrayBuffer()),
      ]);
      engine.session = session; engine.fc = new Float32Array(fcBuf); engine.ready = true;
      el.statArch.textContent = (meta.arch || "—").replace("resnet", "ResNet-");
      el.statClasses.textContent = meta.class_names.length;
      el.statF1.textContent = meta.macro_f1 != null ? Number(meta.macro_f1).toFixed(3) : "—";
      el.brandChip.innerHTML = '<span class="dot"></span> calibrated';
      if (meta.disclaimer) el.disclaimer.textContent = meta.disclaimer;
    } catch (e) {
      console.error("model load failed", e);
      showDemo("Could not load the in-browser model (model/meta.json + model/*.onnx). Run scripts/export_onnx.py and redeploy.");
      el.statArch.textContent = "—"; el.statClasses.textContent = "—"; el.statF1.textContent = "—";
    }
  })();
})();
