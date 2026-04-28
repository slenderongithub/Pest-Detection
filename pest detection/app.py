import streamlit as st
import numpy as np
import torch
from PIL import Image
import importlib
from pathlib import Path
from io import BytesIO
from urllib.request import urlretrieve

# ── PyTorch 2.6 compatibility: fastai v1 pkl needs weights_only=False ──────────
_orig_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafScan — Plant Disease Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Disease knowledge base ─────────────────────────────────────────────────────
DISEASE_INFO = {
    "apple scab": {
        "severity": "Medium",
        "cause": "Fungal (Venturia inaequalis)",
        "tip": "Apply fungicide sprays in early spring. Remove and destroy fallen leaves.",
    },
    "black rot": {
        "severity": "High",
        "cause": "Fungal (Botryosphaeria obtusa)",
        "tip": "Prune infected branches 8–15 cm below visible symptoms. Apply copper-based fungicide.",
    },
    "cedar apple rust": {
        "severity": "Medium",
        "cause": "Fungal (Gymnosporangium juniperi-virginianae)",
        "tip": "Remove nearby cedar/juniper hosts. Apply myclobutanil fungicide preventively.",
    },
    "bacterial spot": {
        "severity": "High",
        "cause": "Bacterial (Xanthomonas spp.)",
        "tip": "Use copper-based bactericide. Avoid overhead irrigation. Remove infected plant debris.",
    },
    "early blight": {
        "severity": "Medium",
        "cause": "Fungal (Alternaria solani)",
        "tip": "Apply chlorothalonil or mancozeb fungicide. Ensure good air circulation between plants.",
    },
    "late blight": {
        "severity": "Critical",
        "cause": "Oomycete (Phytophthora infestans)",
        "tip": "Apply metalaxyl fungicide immediately. Remove and bag infected plants. Avoid overhead watering.",
    },
    "leaf mold": {
        "severity": "Medium",
        "cause": "Fungal (Passalora fulva)",
        "tip": "Improve greenhouse ventilation. Apply copper fungicide. Keep humidity below 85%.",
    },
    "septoria leaf spot": {
        "severity": "Medium",
        "cause": "Fungal (Septoria lycopersici)",
        "tip": "Remove lower infected leaves. Apply mancozeb or chlorothalonil. Rotate crops annually.",
    },
    "powdery mildew": {
        "severity": "Medium",
        "cause": "Fungal (various Erysiphales)",
        "tip": "Apply potassium bicarbonate or sulfur-based fungicide. Improve air circulation.",
    },
    "northern leaf blight": {
        "severity": "High",
        "cause": "Fungal (Exserohilum turcicum)",
        "tip": "Plant resistant hybrids. Apply propiconazole fungicide at first sign of infection.",
    },
    "cercospora leaf spot": {
        "severity": "Medium",
        "cause": "Fungal (Cercospora zeae-maydis)",
        "tip": "Use resistant varieties. Apply strobilurin-based fungicide preventively.",
    },
    "common rust": {
        "severity": "Medium",
        "cause": "Fungal (Puccinia sorghi)",
        "tip": "Plant resistant corn hybrids. Apply fungicide if severity exceeds 5% leaf area.",
    },
    "esca": {
        "severity": "Critical",
        "cause": "Fungal complex (Phaeomoniella, Phaeoacremonium)",
        "tip": "No chemical cure available. Remove infected vines. Protect pruning wounds with fungicide paste.",
    },
    "haunglongbing": {
        "severity": "Critical",
        "cause": "Bacterial (Candidatus Liberibacter)",
        "tip": "No cure exists. Remove infected trees immediately. Control psyllid vector with insecticide.",
    },
    "leaf scorch": {
        "severity": "Medium",
        "cause": "Fungal (Diplocarpon earlianum)",
        "tip": "Apply captan fungicide. Remove infected leaves. Avoid overhead watering.",
    },
    "target spot": {
        "severity": "Medium",
        "cause": "Fungal (Corynespora cassiicola)",
        "tip": "Apply azoxystrobin or chlorothalonil. Reduce canopy humidity through pruning.",
    },
    "yellow leaf curl virus": {
        "severity": "Critical",
        "cause": "Viral (Tomato yellow leaf curl virus)",
        "tip": "No cure. Remove infected plants. Use reflective mulches and insecticides to control whitefly vector.",
    },
    "mosaic virus": {
        "severity": "High",
        "cause": "Viral (Tobamovirus)",
        "tip": "No cure. Wash hands and tools frequently. Remove and destroy infected plants.",
    },
    "spider mites": {
        "severity": "Medium",
        "cause": "Pest (Tetranychus urticae)",
        "tip": "Apply miticide or neem oil. Increase humidity. Introduce predatory mites as biological control.",
    },
}

SEVERITY_CONFIG = {
    "Critical": {"color": "#ff4444", "bg": "rgba(255,68,68,0.1)", "icon": "🔴"},
    "High":     {"color": "#ff8800", "bg": "rgba(255,136,0,0.1)",  "icon": "🟠"},
    "Medium":   {"color": "#ffcc00", "bg": "rgba(255,204,0,0.1)",  "icon": "🟡"},
    "Low":      {"color": "#44cc44", "bg": "rgba(68,204,68,0.1)",  "icon": "🟢"},
    "Healthy":  {"color": "#00e676", "bg": "rgba(0,230,118,0.1)",  "icon": "✅"},
}

def pretty_label(label: str) -> str:
    return label.replace("___", " — ").replace("_", " ").strip()

def get_disease_info(label: str):
    label_lower = label.lower()
    for key, info in DISEASE_INFO.items():
        if key in label_lower:
            return info
    return None

def get_severity(label: str, confidence: float):
    if "healthy" in label.lower():
        return "Healthy"
    info = get_disease_info(label)
    if info:
        return info["severity"]
    # fallback based on confidence
    if confidence >= 85:
        return "High"
    elif confidence >= 60:
        return "Medium"
    return "Low"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #0d1117; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0d2818 0%, #0d1117 50%, #1a0d2e 100%);
    border: 1px solid rgba(0, 230, 118, 0.15);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,230,118,0.05) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(99,102,241,0.05) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00e676, #69f0ae, #a5d6a7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,230,118,0.1);
    border: 1px solid rgba(0,230,118,0.3);
    color: #00e676;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 1rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(22, 27, 34, 0.9);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Result card ── */
.result-disease {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0.5rem 0;
}
.result-confidence {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1;
}

/* ── Severity badge ── */
.severity-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}

/* ── Info row ── */
.info-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    align-items: flex-start;
}
.info-label {
    color: #8b949e;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    min-width: 70px;
    padding-top: 2px;
}
.info-value {
    color: #e6edf3;
    font-size: 0.88rem;
}

/* ── Tip box ── */
.tip-box {
    background: rgba(0,230,118,0.06);
    border-left: 3px solid #00e676;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-top: 1rem;
    font-size: 0.88rem;
    color: #c9d1d9;
    line-height: 1.6;
}
.tip-box b { color: #00e676; }

/* ── Progress bars ── */
.pred-row {
    margin: 0.5rem 0;
}
.pred-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #8b949e;
    margin-bottom: 4px;
}
.pred-bar-bg {
    background: rgba(48,54,61,0.8);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.pred-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* ── Section title ── */
.section-title {
    color: #8b949e;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.75rem;
}

/* ── Scan history ── */
.history-item {
    background: rgba(22,27,34,0.7);
    border: 1px solid rgba(48,54,61,0.6);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #c9d1d9;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* ── Uploader styling ── */
[data-testid="stFileUploader"] {
    background: rgba(22,27,34,0.6);
    border: 2px dashed rgba(0,230,118,0.25);
    border-radius: 14px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,230,118,0.5);
}

/* ── Metric overrides ── */
[data-testid="stMetric"] {
    background: rgba(22,27,34,0.8);
    border: 1px solid rgba(48,54,61,0.6);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.4rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🔬 LeafScan</p>
    <p class="hero-sub">AI-powered plant disease detection · Powered by ResNet34 + PlantVillage</p>
    <span class="hero-badge">38 disease classes · 11 plant species</span>
</div>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
EXPORT_FILE_URL = "https://drive.google.com/uc?export=download&id=1qGyr9AEj71iLITNpjBaKqY7DYJ2xrVfm"
MODEL_DIR  = Path(__file__).resolve().parent / "app" / "models"
MODEL_PATH = MODEL_DIR / "export_resnet34_model.pkl"

def get_fastai_helpers():
    try:
        fv = importlib.import_module("fastai.vision")
        return fv.load_learner, fv.open_image
    except Exception as e:
        st.session_state["fastai_error"] = str(e)
        return None, None

def ensure_model_file():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        urlretrieve(EXPORT_FILE_URL, MODEL_PATH)

@st.cache_resource(show_spinner=False)
def load_model_once():
    load_fn, _ = get_fastai_helpers()
    try:
        ensure_model_file()
    except Exception:
        return None, "missing"
    if not MODEL_PATH.exists():
        return None, "missing"
    if load_fn is None:
        return None, "fastai_missing"
    try:
        learner = load_fn(MODEL_PATH.parent, MODEL_PATH.name)
        return learner, "loaded"
    except Exception:
        return None, "load_failed"

with st.spinner("Loading model…"):
    model, model_status = load_model_once()

# ── Session state: scan history ────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Layout ─────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<p class="section-title">Upload Leaf Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a leaf photo here or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="")

    # ── Stats strip ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "ResNet34")
    m2.metric("Classes", "38")
    m3.metric("Scans", len(st.session_state.history))

with right_col:
    st.markdown('<p class="section-title">Analysis Result</p>', unsafe_allow_html=True)

    if not uploaded_file:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 3rem 1.5rem; color:#8b949e;">
            <div style="font-size:3rem; margin-bottom:1rem;">🍃</div>
            <div style="font-weight:600; font-size:1rem; color:#c9d1d9;">No image uploaded yet</div>
            <div style="font-size:0.85rem; margin-top:0.5rem;">Upload a leaf photo on the left to begin diagnosis</div>
        </div>
        """, unsafe_allow_html=True)

    elif model_status in ("missing", "fastai_missing", "load_failed"):
        msgs = {
            "missing":        "Model file not found in `app/models/`. Place `export_resnet34_model.pkl` there and restart.",
            "fastai_missing": "fastai is not importable. Run `pip install fastai==1.0.61 ipython` and restart Streamlit.",
            "load_failed":    "Model file found but failed to load. Check PyTorch/fastai version compatibility.",
        }
        st.error(f"⚠️ {msgs[model_status]}")

    else:
        with st.spinner("Scanning leaf…"):
            _, open_image_fn = get_fastai_helpers()
            img = open_image_fn(BytesIO(uploaded_file.getvalue()))
            pred_class, _, probabilities = model.predict(img)
            pred_label      = str(pred_class)
            confidence      = round(float(probabilities.max().item()) * 100, 2)
            is_healthy      = "healthy" in pred_label.lower()
            severity        = get_severity(pred_label, confidence)
            sev_cfg         = SEVERITY_CONFIG[severity]
            disease_info    = get_disease_info(pred_label)
            display_name    = pretty_label(pred_label)

            # Save to history
            st.session_state.history.insert(0, {
                "name": display_name, "confidence": confidence, "severity": severity
            })
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[:5]

        # ── Severity badge ────────────────────────────────────────────────────
        st.markdown(f"""
        <span class="severity-badge" style="background:{sev_cfg['bg']}; color:{sev_cfg['color']}; border:1px solid {sev_cfg['color']}40;">
            {sev_cfg['icon']} {severity} Severity
        </span>
        """, unsafe_allow_html=True)

        # ── Disease name + confidence ──────────────────────────────────────────
        conf_color = sev_cfg["color"]
        st.markdown(f"""
        <div class="glass-card">
            <div class="result-disease">{display_name}</div>
            <div class="result-confidence" style="color:{conf_color};">{confidence}%</div>
            <div style="color:#8b949e; font-size:0.8rem; margin-top:0.25rem;">confidence score</div>
        """, unsafe_allow_html=True)

        if disease_info:
            st.markdown(f"""
            <div style="margin-top:1.2rem;">
                <div class="info-row">
                    <span class="info-label">Cause</span>
                    <span class="info-value">{disease_info['cause']}</span>
                </div>
            </div>
            <div class="tip-box">
                <b>💡 Treatment:</b> {disease_info['tip']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Top 3 predictions ─────────────────────────────────────────────────
        st.markdown('<p class="section-title" style="margin-top:1rem;">Top Predictions</p>', unsafe_allow_html=True)
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        class_vocab = getattr(model.data, "classes", [])
        colors = [conf_color, "#6366f1", "#8b949e"]

        for i in range(3):
            lbl  = pretty_label(str(class_vocab[top3_indices[i].item()]) if class_vocab else str(top3_indices[i].item()))
            prob = round(top3_probs[i].item() * 100, 2)
            st.markdown(f"""
            <div class="pred-row">
                <div class="pred-label"><span>{lbl}</span><span style="color:{colors[i]};font-weight:600;">{prob}%</span></div>
                <div class="pred-bar-bg">
                    <div class="pred-bar-fill" style="width:{prob}%; background:{colors[i]};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Scan history ───────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Recent Scans</p>', unsafe_allow_html=True)
    for h in st.session_state.history:
        sev = SEVERITY_CONFIG[h["severity"]]
        st.markdown(f"""
        <div class="history-item">
            <span>{sev['icon']} {h['name']}</span>
            <span style="color:{sev['color']}; font-weight:600;">{h['confidence']}%</span>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:0.75rem; margin-top:3rem; padding-bottom:1rem;">
    LeafScan · ResNet34 · PlantVillage Dataset · VIT-AP University
</div>
""", unsafe_allow_html=True)