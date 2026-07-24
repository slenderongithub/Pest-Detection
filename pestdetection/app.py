"""LeafScan Streamlit UI — a thin shell over the shared ``leafscan`` package.

Editorial "paper + forest" theme matching the Starlette SPA. No ML logic lives here: the
taxonomy, model, Grad-CAM, severity, and pest knowledge all come from ``leafscan``. With no
model installed the app shows an explicit no-model state instead of fabricating a prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from leafscan.inference import Predictor  # noqa: E402
from leafscan.knowledge import disclaimer  # noqa: E402

st.set_page_config(page_title="LeafScan — Pest Scanner", page_icon="🐛", layout="wide",
                   initial_sidebar_state="collapsed")

SEVERITY_COLORS = {
    "Low": "#4b7b58", "Medium": "#a97a1f", "High": "#b4531f", "Critical": "#9a2f24",
}

CUSTOM_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --paper:#f4f1ea; --card:#fbfaf6; --card-2:#f7f4ed; --ink:#1a1c19; --ink-2:#41443e;
  --ink-3:#73766e; --line:#e2ddd1; --accent:#1c4b3c; --accent-soft:#e5ede8; }
html, body, [class*='css'] { font-family:'Inter',sans-serif; }
.stApp { background:
  radial-gradient(circle at top right, rgba(28,75,60,0.06), transparent 30%),
  var(--paper); color:var(--ink); }
#MainMenu, footer, header { visibility:hidden; }
.hero { border-radius:24px; padding:1.7rem 1.9rem; border:1px solid var(--line);
  background:linear-gradient(120deg,var(--card),var(--card-2)); box-shadow:0 12px 34px -18px rgba(26,28,25,.18); }
.eyebrow { display:inline-block; padding:.4rem .8rem; border-radius:999px; background:var(--accent-soft);
  border:1px solid rgba(28,75,60,.2); color:var(--accent); font-size:.68rem; font-weight:700;
  letter-spacing:.16em; text-transform:uppercase; font-family:'JetBrains Mono',monospace; }
.hero-title { margin:.8rem 0 .2rem; font-family:'Fraunces',serif; font-size:clamp(2rem,5vw,3.4rem);
  line-height:1; letter-spacing:-0.03em; color:var(--ink); font-weight:600; }
.hero-title span { color:var(--accent); font-style:italic; }
.hero-sub { color:var(--ink-3); font-size:1rem; max-width:64ch; line-height:1.6; }
.card { border-radius:22px; border:1px solid var(--line); background:var(--card); padding:1.2rem;
  box-shadow:0 12px 34px -20px rgba(26,28,25,.18); }
.label { color:var(--ink-3); font-size:.68rem; font-weight:600; letter-spacing:.16em;
  text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-bottom:.5rem; }
.result-name { font-family:'Fraunces',serif; font-size:clamp(1.4rem,2.6vw,2.1rem); font-weight:600;
  color:var(--ink); margin:.3rem 0 .1rem; }
.result-conf { font-family:'Fraunces',serif; font-size:clamp(2.2rem,5vw,3.4rem); line-height:1;
  font-weight:600; letter-spacing:-0.03em; }
.pill { display:inline-block; padding:.4rem .85rem; border-radius:999px; font-weight:700;
  font-size:.72rem; font-family:'Inter',sans-serif; }
.tip { margin-top:.9rem; padding:.9rem 1rem; border-radius:16px; background:var(--accent-soft);
  border:1px solid rgba(28,75,60,.18); color:var(--ink-2); line-height:1.6; font-size:.9rem; }
.warn { margin-top:.6rem; padding:.9rem 1rem; border-radius:16px;
  background:rgba(169,122,31,.10); border:1px solid rgba(169,122,31,.3); color:#7a5312; }
.mini { color:var(--ink-3); font-size:.8rem; line-height:1.5; }
.ptable { width:100%; border-collapse:collapse; margin-top:.4rem; }
.ptable td { padding:.4rem .5rem; border-bottom:1px solid var(--line); font-size:.9rem; }
.ptable td.dose { color:var(--accent); text-align:right; font-family:'JetBrains Mono',monospace; }
.bar-bg { height:7px; border-radius:999px; background:var(--card-2); border:1px solid var(--line);
  overflow:hidden; margin-bottom:.6rem; }
.bar-fill { height:100%; border-radius:999px; }
[data-testid='stFileUploader'] { background:var(--card-2); border:1.5px dashed rgba(28,75,60,.35);
  border-radius:18px; padding:1rem; }
</style>
"""
st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_predictor() -> Predictor:
    return Predictor()


predictor = get_predictor()
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    """
    <div class="hero">
      <span class="eyebrow">Agricultural pest intelligence</span>
      <div class="hero-title">Leaf<span>Scan</span></div>
      <p class="hero-sub">Upload a photo of a pest on a plant. LeafScan identifies it against its trained
      classes, shows a calibrated confidence, boxes the regions it focused on, and surfaces reference
      treatment guidance. It abstains when unsure and never guesses when no model is loaded.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not predictor.is_available:
    st.markdown(
        f"""
        <div class="card" style="margin-top:1.1rem;border-color:rgba(169,122,31,.4);">
          <div class="label" style="color:#a97a1f;">No model installed — demo state</div>
          <div style="color:var(--ink);font-weight:700;margin-bottom:.4rem;">LeafScan will not fabricate a prediction.</div>
          <div class="mini">No checkpoint at <code>{predictor.model_dir}</code>. Train one:
          <code>leafscan train --config configs/pest_resnet50.yaml</code> — or fetch weights:
          <code>python scripts/fetch_model.py</code>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

bundle = predictor.bundle
cols = st.columns(4, gap="medium")
cols[0].metric("Model", predictor.model_name)
cols[1].metric("Classes", bundle.num_classes)
cols[2].metric("Macro-F1", (bundle.metrics or {}).get("macro_f1", "—"))
cols[3].metric("Recent scans", len(st.session_state.history))

left, right = st.columns([1.0, 1.2], gap="large")

with left:
    st.markdown('<div class="label">Upload pest image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="label">Diagnosis</div>', unsafe_allow_html=True)
    if not uploaded:
        st.markdown('<div class="card" style="min-height:300px;display:flex;align-items:center;'
                    'justify-content:center;text-align:center;"><div class="mini" style="max-width:30ch;">'
                    "Upload an image to get a prediction, calibrated confidence, boxed focus regions, and "
                    "reference treatment guidance.</div></div>", unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing…"):
            pred = predictor.predict(image, want_gradcam=True)
        color = SEVERITY_COLORS.get(pred.severity or "", "#73766e")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if pred.severity:
            st.markdown(f'<span class="pill" style="background:{color}22;color:{color};'
                        f'border:1px solid {color}55;">{pred.severity} severity</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-name">{pred.display_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-conf" style="color:{color};">{pred.confidence:.1f}%</div>'
                    '<div class="mini">calibrated confidence</div>', unsafe_allow_html=True)
        if pred.abstained:
            st.markdown(f'<div class="warn"><strong>Uncertain.</strong> {pred.confidence:.1f}% is below the '
                        f"{pred.abstain_threshold:.0f}% threshold — a weak hint, likely outside the trained "
                        "classes.</div>", unsafe_allow_html=True)
        info = pred.pest_info
        if info:
            st.markdown(f'<div style="margin-top:.7rem;color:var(--ink-2);font-size:.9rem;">'
                        f'<strong>{info["common_name"]}</strong> · {info["pest_type"]}</div>'
                        f'<div class="mini" style="margin-top:.3rem;">{info["description"]}</div>',
                        unsafe_allow_html=True)
            if info["pesticides"]:
                rows = "".join(f'<tr><td>{p["name"]}</td><td class="dose">{p["dose"]}</td></tr>'
                               for p in info["pesticides"])
                st.markdown('<div class="label" style="margin-top:.9rem;">Reference treatment (per L water)</div>'
                            f'<table class="ptable">{rows}</table>', unsafe_allow_html=True)
            if info["ipm"]:
                st.markdown(f'<div class="tip"><strong>IPM:</strong> {info["ipm"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown('<div class="card"><div class="label">Focus regions</div>', unsafe_allow_html=True)
            st.image(pred.overlay if pred.overlay is not None else image, use_container_width=True)
            if pred.overlay is None:
                st.caption("Abstained — no region drawn." if pred.abstained else "No strong region highlighted.")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><div class="label">Top predictions</div>', unsafe_allow_html=True)
            palette = [color, "#1c4b3c", "#73766e"]
            for i, tp in enumerate(pred.top_predictions):
                w = tp["probability"]
                st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:.85rem;'
                            f'color:var(--ink-2);margin-bottom:.25rem;"><span>{tp["display_label"]}</span>'
                            f'<span style="color:{palette[i % 3]};font-weight:700;">{w:.1f}%</span></div>'
                            f'<div class="bar-bg"><div class="bar-fill" style="width:{w}%;'
                            f'background:{palette[i % 3]};"></div></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.history.insert(0, {"name": pred.display_label, "confidence": pred.confidence,
                                            "severity": pred.severity})
        st.session_state.history = st.session_state.history[:5]

if st.session_state.history:
    st.markdown('<div class="label" style="margin-top:1rem;">Recent scans</div>', unsafe_allow_html=True)
    for item in st.session_state.history:
        c = SEVERITY_COLORS.get(item["severity"] or "", "#73766e")
        st.markdown(f'<div style="display:flex;justify-content:space-between;padding:.7rem 1rem;'
                    f'border-radius:14px;background:var(--card);border:1px solid var(--line);margin-bottom:.5rem;">'
                    f'<span style="color:var(--ink);font-weight:700;">{item["name"]}</span>'
                    f'<span style="color:{c};font-weight:800;font-family:JetBrains Mono,monospace;">'
                    f'{item["confidence"]:.1f}%</span></div>', unsafe_allow_html=True)

st.markdown(f'<div style="color:var(--ink-3);font-size:.74rem;padding:1.3rem 0;line-height:1.5;">{disclaimer()}</div>',
            unsafe_allow_html=True)
