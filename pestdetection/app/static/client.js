(() => {
  "use strict";
  const RING_CIRC = 2 * Math.PI * 52; // r=52 in the SVG
  const FRAME_MS = 1200;

  const $ = (id) => document.getElementById(id);
  const el = {
    root: document.documentElement,
    themeToggle: $("theme-toggle"),
    brandChip: $("brand-chip"),
    demoBanner: $("demo-banner"),
    demoMsg: $("demo-msg"),
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
    applyTheme(next);
    localStorage.setItem("leafscan-theme", next);
  });

  /* ---------- demo / no-model ---------- */
  function showDemo(message) {
    state.demo = true;
    el.demoBanner.hidden = false;
    if (message) el.demoMsg.textContent = message;
    el.brandChip.innerHTML = '<span class="dot" style="background:var(--amber)"></span> demo mode';
    el.startCam.disabled = true;
    el.capture.disabled = true;
    el.dropTitle.textContent = "No model installed";
    el.drop.style.pointerEvents = "none";
    el.drop.style.opacity = ".55";
  }

  /* ---------- rendering ---------- */
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
    if (!info) {
      el.pestBody.hidden = true; el.pestEmpty.hidden = false; return;
    }
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
    const color = sevVar(r.severity);
    state.history.unshift({
      label: r.display_label || r.label || "Unknown",
      pct: Number(r.confidence ?? 0),
      sev: r.severity,
      abstained: r.abstained,
    });
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
    void color;
  }

  function renderFocus(result) {
    if (result.overlay_data_url) {
      el.focusImg.src = result.overlay_data_url;
      el.focusImg.hidden = false;
      el.focusPlaceholder.hidden = true;
    }
    const n = (result.boxes || []).length;
    el.focusRegions.hidden = false;
    el.focusRegions.textContent = n === 1 ? "1 region" : `${n} regions`;
  }

  function renderResult(result, withFocus) {
    el.resultEmpty.hidden = true;
    el.resultBody.hidden = false;
    el.resultChip.hidden = false;

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

    renderPest(result.pest_info);
    renderTreatment(result.pest_info);
    renderTop3(result.top_predictions);
    if (withFocus) renderFocus(result);
    if (result.disclaimer) el.disclaimer.textContent = result.disclaimer;
    pushHistory(result);
  }

  /* ---------- network ---------- */
  async function analyze(blob, endpoint, withFocus) {
    if (state.busy || state.demo) return;
    state.busy = true;
    try {
      const fd = new FormData();
      fd.append("file", blob, "scan.jpg");
      const res = await fetch(endpoint, { method: "POST", body: fd });
      if (res.status === 503) {
        const body = await res.json().catch(() => ({}));
        showDemo(body.message);
        return;
      }
      if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b.error || `HTTP ${res.status}`); }
      renderResult(await res.json(), withFocus);
    } catch (e) {
      console.error("analyze failed", e);
    } finally {
      state.busy = false;
    }
  }

  /* ---------- upload ---------- */
  el.fileInput.addEventListener("change", () => {
    const f = el.fileInput.files && el.fileInput.files[0];
    if (f) { el.dropTitle.textContent = f.name.slice(0, 28); analyze(f, "/analyze", true); }
  });
  ["dragover", "dragenter"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => el.drop.addEventListener(ev, () => el.drop.classList.remove("drag")));
  el.drop.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) { el.dropTitle.textContent = f.name.slice(0, 28); analyze(f, "/analyze", true); }
  });

  /* ---------- camera ---------- */
  function captureFrame(endpoint, withFocus) {
    if (!state.stream || !el.camFeed.videoWidth) return;
    const c = el.canvas;
    c.width = el.camFeed.videoWidth; c.height = el.camFeed.videoHeight;
    const ctx = c.getContext("2d");
    ctx.save(); ctx.scale(-1, 1); ctx.drawImage(el.camFeed, -c.width, 0, c.width, c.height); ctx.restore();
    c.toBlob((b) => b && analyze(b, endpoint, withFocus), "image/jpeg", 0.9);
  }

  async function startCamera() {
    if (state.demo) return;
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false,
      });
      el.camFeed.srcObject = state.stream;
      await el.camFeed.play();
      el.camStage.classList.add("on");
      el.scanline.hidden = false; el.camRec.hidden = false;
      el.startCam.hidden = true; el.capture.hidden = false; el.stopCam.hidden = false;
      el.camStatus.innerHTML = '<span class="live-dot"></span> live';
      state.recStart = 0;
      state.rec = setInterval(() => {
        state.recStart++;
        const m = String(Math.floor(state.recStart / 60)).padStart(2, "0");
        const s = String(state.recStart % 60).padStart(2, "0");
        el.camRec.innerHTML = `<span class="live-dot"></span> REC ${m}:${s}`;
      }, 1000);
      state.timer = setInterval(() => captureFrame("/analyze-frame", false), FRAME_MS);
      captureFrame("/analyze-frame", false);
    } catch (e) {
      el.camStatus.innerHTML = '<span class="dot" style="background:var(--sev-high)"></span> unavailable';
      console.error("camera error", e);
    }
  }

  function stopCamera() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    if (state.rec) { clearInterval(state.rec); state.rec = null; }
    if (state.stream) { state.stream.getTracks().forEach((t) => t.stop()); state.stream = null; }
    el.camFeed.srcObject = null;
    el.camStage.classList.remove("on");
    el.scanline.hidden = true; el.camRec.hidden = true;
    el.startCam.hidden = false; el.capture.hidden = true; el.stopCam.hidden = true;
    el.camStatus.innerHTML = '<span class="dot"></span> idle';
  }

  el.startCam.addEventListener("click", startCamera);
  el.stopCam.addEventListener("click", stopCamera);
  el.capture.addEventListener("click", () => captureFrame("/analyze", true));
  window.addEventListener("beforeunload", stopCamera);

  /* ---------- boot ---------- */
  (async function boot() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (!data.model_available) { showDemo(); el.statArch.textContent = "—"; el.statClasses.textContent = "—"; el.statF1.textContent = "—"; return; }
      el.statArch.textContent = (data.arch || data.model || "—").replace("resnet", "ResNet-");
      el.statClasses.textContent = data.class_count != null ? data.class_count : "—";
      el.statF1.textContent = data.macro_f1 != null ? Number(data.macro_f1).toFixed(3) : "—";
    } catch (e) {
      el.statArch.textContent = "offline"; console.warn("health failed", e);
    }
  })();
})();
