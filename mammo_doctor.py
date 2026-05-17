"""
MammoDoctor — Professional Breast Cancer Detection Platform
===========================================================
Five-stage AI system for radiologists and oncologists.
NOVEL: First calibrated uncertainty-aware multi-modal mammography ensemble
       with model-disagreement detection and NL clinical explanations.

Models:
  Stage 1  gb_model.pkl              — CBIS-DDSM only   (AUC 0.8678)
  Stage 1+ gb_multi_dataset.pkl     — CBIS + VinDr      (AUC 0.9925)
  Stage 2  best_model.pth           — EfficientNet-B4   (Sens 87.3%)
  Sub-class gb_benign_subclass.pkl  — biopsy need       (AUC 0.9729)
  Ensemble mammo_ensemble.pkl       — calibrated+uncert (ECE 0.052)

Launch:
    /opt/anaconda3/envs/aicd/bin/streamlit run mammo_doctor.py
"""

import io, sys, json, pickle, warnings, base64, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageEnhance

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
MODELS   = ROOT / "models"
CKPT     = ROOT / "stage2_cnn" / "checkpoints" / "best_model.pth"
RESULTS  = ROOT / "results"
sys.path.insert(0, str(ROOT / "stage2_cnn"))

# ── optional imports ───────────────────────────────────────────────────────
try:
    import pydicom
    DICOM_OK = True
except ImportError:
    DICOM_OK = False

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    from skimage import exposure as skexp
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

# ── calibration ensemble (local module) ────────────────────────────────────
sys.path.insert(0, str(ROOT / "src"))
try:
    from calibration_ensemble import (
        MammoEnsemble, run_mc_dropout, enable_mc_dropout,
        generate_clinical_explanation, ece,
    )
    ENSEMBLE_OK = True
except ImportError:
    ENSEMBLE_OK = False

# ── feature definitions ────────────────────────────────────────────────────
FEATURE_COLS = [
    "assessment", "subtlety", "breast_density", "is_mass",
    "calc_type_risk", "calc_dist_risk", "mass_shape_risk", "mass_margin_risk",
    "morph_risk", "view_mlo", "is_right",
]
FEATURE_NAMES = [
    "BI-RADS Assessment", "Subtlety", "Breast Density", "Is Mass",
    "Calc Type Risk", "Calc Dist Risk", "Mass Shape Risk", "Mass Margin Risk",
    "Morphology Risk", "View MLO", "Right Breast",
]

CALC_TYPE_RISK  = {"PLEOMORPHIC":3,"AMORPHOUS":2,"HETEROGENEOUS":2,"FINE_LINEAR_BRANCHING":3,
                   "PUNCTATE":1,"ROUND_AND_REGULAR":0,"COARSE":0,"MILK_OF_CALCIUM":0,"N/A":1}
CALC_DIST_RISK  = {"CLUSTERED":2,"LINEAR":3,"SEGMENTAL":2,"REGIONAL":1,"DIFFUSELY_SCATTERED":0}
MASS_SHAPE_RISK = {"IRREGULAR":3,"IRREGULAR-ARCHITECTURAL_DISTORTION":3,"LOBULATED":2,
                   "OVAL":1,"ROUND":1,"ARCHITECTURAL_DISTORTION":2}
MASS_MARGIN_RISK= {"SPICULATED":3,"ILL_DEFINED":2,"OBSCURED":1,"MICROLOBULATED":2,"CIRCUMSCRIBED":0}

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MammoDoctor — AI Mammography Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ══════════════════════════════════════════
   GLOBAL — clean white canvas, sharp text
   ══════════════════════════════════════════ */
html, body, [data-testid="stAppViewContainer"] {
    background: #ffffff !important;
    color: #111827 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stMain"] { background: #ffffff !important; }
[data-testid="block-container"] { padding-top: 1rem !important; }

/* Every Streamlit text element — force dark */
p, span, label, div, h1, h2, h3, h4, li, td, th, .stMarkdown {
    color: #111827 !important;
}

/* ══════════════════════════════════════════
   SIDEBAR — deep navy, crisp white text
   ══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0f2347 100%) !important;
    border-right: 3px solid #2563eb !important;
}
[data-testid="stSidebar"] * { color: #f0f6ff !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select {
    background: #1e3a6e !important;
    color: #ffffff !important;
    border: 1px solid #3b7dd8 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] hr { border-color: #2563eb !important; opacity: 0.4; }
[data-testid="stSidebar"] .stFileUploader {
    background: #1e3a6e !important;
    border: 2px dashed #3b7dd8 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stFileUploader * { color: #cce0ff !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #60a5fa !important;
    font-weight: 700 !important;
    border-bottom: 1px solid #2563eb;
    padding-bottom: 4px;
}

/* ══════════════════════════════════════════
   TAB BAR
   ══════════════════════════════════════════ */
[data-testid="stTabs"] [role="tablist"] {
    background: #f1f5f9;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #374151 !important;
    padding: 0.5rem 1.1rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
}

/* ══════════════════════════════════════════
   HEADER BANNER
   ══════════════════════════════════════════ */
.main-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #3b82f6 100%);
    padding: 1.8rem 2.4rem;
    border-radius: 18px;
    margin-bottom: 1.4rem;
    color: #ffffff !important;
    box-shadow: 0 6px 30px rgba(37,99,235,0.35);
    border: none;
}
.main-header h1 {
    margin: 0; font-size: 2rem; font-weight: 900;
    letter-spacing: -0.8px; color: #ffffff !important;
}
.main-header p {
    margin: 0.4rem 0 0; font-size: 0.9rem;
    color: rgba(255,255,255,0.85) !important;
}

/* ══════════════════════════════════════════
   RISK CARDS — vivid, high-contrast
   ══════════════════════════════════════════ */
.risk-card {
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    text-align: center;
    font-weight: 900;
    font-size: 1.6rem;
    letter-spacing: 0.3px;
    margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.18);
}
.risk-low {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    color: #064e3b !important;
    border: 2.5px solid #10b981;
}
.risk-medium {
    background: linear-gradient(135deg, #fef9c3, #fde68a);
    color: #78350f !important;
    border: 2.5px solid #f59e0b;
}
.risk-high {
    background: linear-gradient(135deg, #fee2e2, #fca5a5);
    color: #7f1d1d !important;
    border: 2.5px solid #ef4444;
}
.risk-veryhigh {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    color: #fff !important;
    border: 2.5px solid #dc2626;
    text-shadow: 0 1px 4px rgba(0,0,0,0.4);
}

/* ══════════════════════════════════════════
   METRIC BOXES — white with bold numbers
   ══════════════════════════════════════════ */
.metric-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    border: 2px solid #e2e8f0;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    transition: transform 0.15s;
}
.metric-box:hover { transform: translateY(-2px); }
.metric-box .val {
    font-size: 1.85rem;
    font-weight: 900;
    color: #1e3a8a !important;
    line-height: 1.1;
}
.metric-box .lbl {
    font-size: 0.7rem;
    color: #4b5563 !important;
    margin-top: 4px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ══════════════════════════════════════════
   STAGE PILLS
   ══════════════════════════════════════════ */
.stage-pill {
    display: inline-block;
    padding: 0.32rem 0.95rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 800;
    margin: 0.2rem 0.25rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.stage-1 { background: #dbeafe; color: #1e3a8a !important; border: 1.5px solid #93c5fd; }
.stage-2 { background: #f3e8ff; color: #581c87 !important; border: 1.5px solid #c084fc; }
.stage-3 { background: #dcfce7; color: #14532d !important; border: 1.5px solid #86efac; }

/* ══════════════════════════════════════════
   CONTENT CARDS
   ══════════════════════════════════════════ */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    border: 2px solid #e2e8f0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
    margin-bottom: 1.1rem;
}
.card-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #1e3a8a !important;
    margin-bottom: 0.9rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #dbeafe;
}

/* ══════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════ */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ══════════════════════════════════════════
   SELECTBOX / INPUTS — full override
   ══════════════════════════════════════════ */

/* --- Labels above every widget --- */
.stSelectbox label, .stSlider label,
.stRadio label, .stTextInput label,
.stNumberInput label,
.stCheckbox label, .stFileUploader label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {
    color: #1e3a8a !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
}

/* --- Selectbox trigger box (the visible closed state) --- */
.stSelectbox [data-baseweb="select"] > div:first-child,
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] > div > div {
    background: #eff6ff !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 8px !important;
    color: #1e3a8a !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    min-height: 44px !important;
}

/* --- Text inside the closed selectbox --- */
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div[class*="ValueContainer"] span,
.stSelectbox [data-baseweb="select"] div[class*="singleValue"],
[data-testid="stSelectbox"] span {
    color: #1e3a8a !important;
    font-weight: 600 !important;
}

/* --- Dropdown arrow icon --- */
.stSelectbox [data-baseweb="select"] svg {
    fill: #2563eb !important;
    color: #2563eb !important;
}

/* ═══════════════════════════════════════════════════════════════
   DROPDOWN POPUP — nuclear white override
   Streamlit renders this in a portal outside the main DOM tree,
   so we need body-level selectors with !important on everything.
   ═══════════════════════════════════════════════════════════════ */

/* Outer popup shell */
body [data-baseweb="popover"],
body [data-baseweb="popover"] > div,
body [data-baseweb="popover"] > div > div {
    background-color: #ffffff !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(37,99,235,0.22) !important;
    overflow: hidden !important;
}

/* Menu list container */
body [data-baseweb="menu"],
body ul[data-baseweb="menu"],
body [role="listbox"],
body [data-baseweb="popover"] ul,
body [data-baseweb="popover"] [role="listbox"] {
    background-color: #ffffff !important;
    padding: 4px !important;
}

/* EVERY descendant inside the popup — force white bg + dark text */
body [data-baseweb="popover"] * {
    background-color: #ffffff !important;
    color: #111827 !important;
}

/* Each individual option row */
body [data-baseweb="menu"] li,
body [data-baseweb="menu"] [role="option"],
body [role="listbox"] [role="option"],
body [data-baseweb="popover"] li {
    background-color: #ffffff !important;
    color: #111827 !important;
    font-weight: 600 !important;
    font-size: 0.93rem !important;
    padding: 10px 16px !important;
    border-radius: 6px !important;
    margin: 2px 4px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    cursor: pointer !important;
}

/* Hover */
body [data-baseweb="menu"] li:hover,
body [data-baseweb="menu"] [role="option"]:hover,
body [role="listbox"] [role="option"]:hover,
body [data-baseweb="popover"] li:hover {
    background-color: #dbeafe !important;
    color: #1e3a8a !important;
}

/* Currently highlighted (keyboard nav) */
body [data-baseweb="menu"] li[aria-selected="true"],
body [role="listbox"] [role="option"][aria-selected="true"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
}

body [data-baseweb="menu"] li[data-highlighted="true"],
body [data-baseweb="popover"] li[data-highlighted="true"] {
    background-color: #bfdbfe !important;
    color: #1e3a8a !important;
}

/* Radio button — fix dark circle */
.stRadio [data-testid="stMarkdownContainer"] p,
.stRadio label span {
    color: #1e3a8a !important;
    font-weight: 600 !important;
}
[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] {
    background: transparent !important;
}
/* Radio unchecked circle */
[data-testid="stRadio"] input[type="radio"] + div,
[data-testid="stRadio"] label > div:first-child {
    border-color: #3b82f6 !important;
    background: #ffffff !important;
}
/* Radio checked fill */
[data-testid="stRadio"] input[type="radio"]:checked + div,
[data-testid="stRadio"] label[data-checked="true"] > div:first-child {
    background: #2563eb !important;
    border-color: #2563eb !important;
}

/* --- Number / text inputs in the main area --- */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #eff6ff !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 8px !important;
    color: #1e3a8a !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #1d4ed8 !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.2) !important;
    outline: none !important;
}

/* --- Slider --- */
[data-testid="stSlider"] > div > div > div > div {
    background: #2563eb !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #2563eb !important;
    border-color: #1d4ed8 !important;
}

/* --- Radio buttons --- */
.stRadio [data-testid="stMarkdownContainer"] p {
    color: #1e3a8a !important;
    font-weight: 600 !important;
}

/* --- Multiselect tags --- */
[data-baseweb="tag"] {
    background: #2563eb !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}

/* ══════════════════════════════════════════
   ALERTS / INFO BOXES
   ══════════════════════════════════════════ */
.stAlert { border-radius: 10px !important; font-weight: 600 !important; }
[data-testid="stSuccess"]  { background: #f0fdf4 !important; border-left: 5px solid #22c55e !important; }
[data-testid="stError"]    { background: #fef2f2 !important; border-left: 5px solid #ef4444 !important; }
[data-testid="stWarning"]  { background: #fffbeb !important; border-left: 5px solid #f59e0b !important; }
[data-testid="stInfo"]     { background: #eff6ff !important; border-left: 5px solid #3b82f6 !important; }
.stAlert p { color: #111827 !important; }

/* ══════════════════════════════════════════
   EXPANDERS & DATA TABLES
   ══════════════════════════════════════════ */
div[data-testid="stExpander"] {
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    background: #f8faff !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #1e3a8a !important;
}
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ══════════════════════════════════════════
   DISCLAIMER / WARNING BOX
   ══════════════════════════════════════════ */
.disclaimer {
    background: #fffbeb;
    border-left: 5px solid #f59e0b;
    padding: 0.9rem 1.2rem;
    border-radius: 10px;
    font-size: 0.82rem;
    color: #451a03 !important;
    font-weight: 500;
    margin-top: 1rem;
}

/* ══════════════════════════════════════════
   SCROLLBAR — subtle blue
   ══════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #93c5fd; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_gb_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_cnn_model():
    if not TORCH_OK:
        return None, None
    if not CKPT.exists():
        return None, None
    try:
        from model import MammoNet, load_checkpoint
        import config as cfg
        device = cfg.get_device()
        model, _ = load_checkpoint(str(CKPT), device)
        model.eval()
        return model, device
    except Exception as e:
        return None, None


@st.cache_resource
def load_benign_model():
    p = MODELS / "gb_benign_subclass.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_ensemble():
    p = MODELS / "mammo_ensemble.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)   # plain dict — no custom class issues


@st.cache_resource
def load_ensemble_metrics():
    p = MODELS / "ensemble_metrics.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _temp_scale(prob: float, T: float) -> float:
    logit  = np.log(np.clip(prob, 1e-8, 1 - 1e-8) /
                    np.clip(1 - prob, 1e-8, 1 - 1e-8))
    scaled = logit / T
    return float(1.0 / (1.0 + np.exp(-scaled)))


def ensemble_predict(bundle: dict,
                     X_features: np.ndarray | None = None,
                     cnn_prob: float | None = None,
                     mc_uncertainty: dict | None = None) -> dict:
    """
    Pure-function ensemble prediction — no custom class needed.
    bundle = plain dict saved by calibration_ensemble.py
    """
    calib_probs = {}

    if X_features is not None:
        X = np.atleast_2d(X_features)
        # CBIS GB
        p_cbis = float(bundle["gb_cbis_pipeline"].predict_proba(X)[0, 1])
        calib_probs["GB (CBIS)"] = _temp_scale(p_cbis, bundle["T_cbis"])
        # Multi-dataset GB
        import pandas as _pd
        X_df = _pd.DataFrame(X, columns=bundle["feature_cols"])
        p_multi = float(bundle["gb_multi_pipeline"].predict_proba(X_df)[0, 1])
        calib_probs["GB (Multi-Dataset)"] = _temp_scale(p_multi, bundle["T_multi"])

    if cnn_prob is not None:
        calib_probs["CNN"] = float(cnn_prob)   # CNN is already a mean of TTA passes

    if not calib_probs:
        raise ValueError("Provide X_features or cnn_prob")

    # Weighted ensemble
    w_map = {
        "GB (CBIS)":        bundle.get("w_cbis", 0.0),
        "GB (Multi-Dataset)": bundle.get("w_multi", 1.0),
        "CNN":              0.30 if cnn_prob is not None else 0.0,
    }
    total_w = sum(w_map.get(k, 0) for k in calib_probs)
    if total_w < 1e-9:
        total_w = len(calib_probs)
        w_map = {k: 1.0 for k in calib_probs}
    ensemble = sum(w_map.get(k, 0) * v for k, v in calib_probs.items()) / total_w

    # Uncertainty
    if mc_uncertainty:
        unc = mc_uncertainty
    else:
        p_arr = np.array(list(calib_probs.values()))
        m = np.clip(ensemble, 1e-8, 1 - 1e-8)
        unc = {"mean": float(p_arr.mean()), "std": float(p_arr.std()),
               "ci_low": float(max(0, ensemble - 2 * p_arr.std())),
               "ci_high": float(min(1, ensemble + 2 * p_arr.std())),
               "entropy": float(-(m * np.log(m) + (1 - m) * np.log(1 - m)))}

    # Model disagreement
    vals = list(calib_probs.values())
    gap  = max(vals) - min(vals) if len(vals) >= 2 else 0.0
    disagree = gap > 0.35

    # BI-RADS mapping
    if ensemble >= 0.85:   birads = "BI-RADS 5 — Highly Suggestive of Malignancy"
    elif ensemble >= 0.65: birads = "BI-RADS 4C — High Suspicion"
    elif ensemble >= 0.45: birads = "BI-RADS 4B — Moderate Suspicion"
    elif ensemble >= 0.25: birads = "BI-RADS 4A — Low Suspicion"
    elif ensemble >= 0.10: birads = "BI-RADS 3 — Probably Benign"
    else:                  birads = "BI-RADS 1/2 — Benign / Negative"

    # NL explanation
    if ENSEMBLE_OK:
        explanation = generate_clinical_explanation(
            ensemble_prob=ensemble, uncertainty=unc,
            model_probs=calib_probs,
            cnn_detected=(cnn_prob >= 0.39) if cnn_prob is not None else None,
        )
    else:
        explanation = _fallback_explanation(ensemble, unc, disagree)

    return {"ensemble_prob": ensemble, "calibrated_probs": calib_probs,
            "uncertainty": unc, "model_disagreement": disagree,
            "disagreement_gap": gap, "birads_category": birads,
            "explanation": explanation}


def _fallback_explanation(prob: float, unc: dict, disagree: bool) -> str:
    lines = []
    if prob >= 0.80:   lines.append(f"HIGHLY SUSPICIOUS — ensemble probability {prob:.0%}.")
    elif prob >= 0.60: lines.append(f"SUSPICIOUS — ensemble probability {prob:.0%}.")
    elif prob >= 0.30: lines.append(f"PROBABLY BENIGN — ensemble probability {prob:.0%}.")
    else:              lines.append(f"BENIGN APPEARANCE — ensemble probability {prob:.0%}.")
    std = unc.get("std", 0)
    ci_lo = unc.get("ci_low", prob); ci_hi = unc.get("ci_high", prob)
    if std > 0.15:
        lines.append(f"⚠ HIGH UNCERTAINTY (σ={std:.3f}, CI [{ci_lo:.0%}–{ci_hi:.0%}]).")
    else:
        lines.append(f"Uncertainty: σ={std:.3f}, CI [{ci_lo:.0%}–{ci_hi:.0%}].")
    if disagree:
        lines.append("⚠ MODEL DISAGREEMENT — radiologist review advised.")
    if prob >= 0.80:   lines.append("RECOMMENDATION: Biopsy indicated.")
    elif prob >= 0.60: lines.append("RECOMMENDATION: Short-interval follow-up or biopsy.")
    elif prob >= 0.30: lines.append("RECOMMENDATION: 6-month follow-up mammogram.")
    else:              lines.append("RECOMMENDATION: Routine annual screening.")
    return "\n\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING
# ════════════════════════════════════════════════════════════════════════════
def read_image(uploaded_file) -> np.ndarray | None:
    name = uploaded_file.name.lower()
    if name.endswith(".dcm"):
        if not DICOM_OK:
            st.error("pydicom not installed — run: pip install pydicom")
            return None
        ds  = pydicom.dcmread(uploaded_file)
        arr = ds.pixel_array.astype(float)
        arr = ((arr - arr.min()) / (arr.ptp() + 1e-8) * 255).astype(np.uint8)
    else:
        arr = np.array(Image.open(uploaded_file).convert("L"))
    return arr


def apply_clahe(arr: np.ndarray) -> np.ndarray:
    if SKIMAGE_OK:
        eq = skexp.equalize_adapthist(arr, clip_limit=0.03)
        return (eq * 255).astype(np.uint8)
    return arr


def preprocess_for_cnn(arr: np.ndarray, device) -> "torch.Tensor":
    """Apply the same preprocessing pipeline as Stage 2 training."""
    clahe_arr = apply_clahe(arr)
    pil = Image.fromarray(clahe_arr).resize((512, 512), Image.LANCZOS).convert("RGB")
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return tfm(pil).unsqueeze(0).to(device)


# ════════════════════════════════════════════════════════════════════════════
# CNN INFERENCE + TTA
# ════════════════════════════════════════════════════════════════════════════
def run_cnn_tta(model, arr: np.ndarray, device) -> tuple[float, list[float]]:
    """5-pass TTA: original + H-flip + V-flip + ±10° rotation."""
    import torchvision.transforms.functional as TF

    def _infer(img_arr):
        t = preprocess_for_cnn(img_arr, device)
        with torch.no_grad():
            logit = model(t)
        return torch.sigmoid(logit).item()

    clahe = apply_clahe(arr)
    pil   = Image.fromarray(clahe).resize((512, 512), Image.LANCZOS)
    passes = [
        np.array(pil), np.array(TF.hflip(pil)), np.array(TF.vflip(pil)),
        np.array(TF.rotate(pil, 10)), np.array(TF.rotate(pil, -10)),
    ]
    probs = [_infer(p) for p in passes]
    return float(np.mean(probs)), probs


def run_cnn_mc_dropout(model, arr: np.ndarray, device, n_passes: int = 50) -> dict:
    """50-pass Monte Carlo Dropout for uncertainty estimation."""
    if ENSEMBLE_OK:
        tensor = preprocess_for_cnn(arr, device)
        return run_mc_dropout(model, tensor, device, n_passes)
    # Fallback: use TTA passes as proxy
    _, tta = run_cnn_tta(model, arr, device)
    p = np.array(tta)
    mean = float(p.mean()); std = float(p.std())
    m = np.clip(mean, 1e-8, 1 - 1e-8)
    return {"mean": mean, "std": std,
            "ci_low": float(max(0, mean - 2*std)),
            "ci_high": float(min(1, mean + 2*std)),
            "entropy": float(-(m*np.log(m) + (1-m)*np.log(1-m))),
            "mutual_info": 0.0, "passes": tta}


# ════════════════════════════════════════════════════════════════════════════
# GRADCAM
# ════════════════════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, out):  self.activations = out.detach()
    def _bwd(self, _, __, go):   self.gradients   = go[0].detach()

    def __call__(self, img_tensor: "torch.Tensor") -> np.ndarray:
        self.model.eval()
        # detach() makes a new leaf tensor — then we can enable grad cleanly
        img = img_tensor.detach().clone()
        with torch.enable_grad():
            logit = self.model(img)
            self.model.zero_grad()
            logit.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        rng = float(cam.max() - cam.min())          # numpy 2.0: no .ptp()
        cam = (cam - cam.min()) / (rng + 1e-8)
        return cam


@st.cache_data(show_spinner=False)
def compute_gradcam(_model, arr_bytes: bytes, _device) -> np.ndarray | None:
    """Cached GradCAM — key on image bytes so different uploads recompute."""
    if _model is None:
        return None
    arr = np.array(Image.open(io.BytesIO(arr_bytes)).convert("L"))
    target = _model.backbone.features[-1]
    gcam   = GradCAM(_model, target)
    tensor = preprocess_for_cnn(arr, _device)   # non-leaf after transforms
    # detach inside GradCAM.__call__ so no requires_grad error here
    cam = gcam(tensor)
    return cam


def overlay_gradcam(orig: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    from PIL import ImageFilter
    h, w = orig.shape[:2]
    cam_up = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))
    cam_up = cam_up / 255.0
    colormap = cm.get_cmap("jet")(cam_up)[..., :3]   # H×W×3
    orig_rgb = np.stack([orig, orig, orig], axis=-1) / 255.0
    blended  = (1 - alpha) * orig_rgb + alpha * colormap
    return (blended * 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# SHAP FOR SINGLE PREDICTION
# ════════════════════════════════════════════════════════════════════════════
def shap_waterfall_for_row(model, row_df: pd.DataFrame, X_train_sample: np.ndarray) -> io.BytesIO | None:
    if not SHAP_OK:
        return None
    try:
        clf = model.named_steps.get("clf")
        imp = model.named_steps.get("imputer") or model.named_steps.get("imp")
        scl = model.named_steps.get("scaler") or model.named_steps.get("scl")
        # Reconstruct preprocessed X
        X_arr = row_df[FEATURE_COLS].values.astype(float) if isinstance(row_df, pd.DataFrame) else row_df
        X_s   = scl.transform(imp.transform(X_arr))
        # We need background data — use training-like data from model internals
        # Just use a sample of zeros as background (fast approximation)
        bg    = np.zeros((50, X_arr.shape[1]))
        bg_s  = scl.transform(imp.transform(bg))
        explainer   = shap.TreeExplainer(clf, bg_s)
        shap_values = explainer.shap_values(X_s)
        base_val    = float(np.array(explainer.expected_value).ravel()[0])
        expl = shap.Explanation(
            values=shap_values[0].astype(float),
            base_values=base_val,
            data=X_s[0].astype(float),
            feature_names=FEATURE_NAMES,
        )
        shap.waterfall_plot(expl, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return buf
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# GAUGE WIDGET
# ════════════════════════════════════════════════════════════════════════════
def render_gauge(prob: float) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(4, 2.3), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    thetas = np.linspace(np.pi, 0, 200)
    cmap   = cm.get_cmap("RdYlGn_r")
    for i in range(len(thetas) - 1):
        ax.plot(thetas[i:i+2], [0.85]*2, color=cmap(i/(len(thetas)-1)), lw=9, solid_capstyle="round")
    needle = np.pi + (0 - np.pi) * prob
    ax.annotate("", xy=(needle, 0.72), xytext=(needle, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="#0f2044", lw=2.8, mutation_scale=20))
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.text(0, -0.18, f"{prob:.1%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#0f2044", transform=ax.transData)
    ax.text(np.pi*1.08, 0.97, "0%",   ha="center", fontsize=7, color="#666")
    ax.text(0,          0.97, "100%", ha="center", fontsize=7, color="#666")
    ax.text(np.pi/2,   1.07, "50%",  ha="center", fontsize=7, color="#666")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def prob_bar(prob: float, label="Malignancy Probability"):
    col = ("#27ae60" if prob < 0.30 else "#f39c12" if prob < 0.60 else
           "#e74c3c" if prob < 0.80 else "#6f1023")
    pct = int(prob * 100)
    st.markdown(f"""
    <div style="background:#eef0f5;border-radius:8px;height:28px;overflow:hidden;margin:4px 0 12px;">
      <div style="width:{pct}%;background:{col};height:100%;border-radius:8px;
                  display:flex;align-items:center;justify-content:center;
                  color:white;font-weight:700;font-size:0.86rem;">
        {pct}%
      </div>
    </div><div style="font-size:0.72rem;color:#888;margin-top:-8px;">{label}</div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATION (matplotlib-based, no external deps)
# ════════════════════════════════════════════════════════════════════════════
def generate_report(patient: dict, cnn_result: dict | None,
                    gb_result: dict | None, sub_result: dict | None,
                    img_arr: np.ndarray | None, cam: np.ndarray | None) -> io.BytesIO:
    """Build a matplotlib multi-page PDF report."""
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ── Page 1: Header + Summary ──────────────────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))  # A4

        # Header bar
        ax_hdr = fig.add_axes([0, 0.88, 1, 0.12])
        ax_hdr.set_facecolor("#0f2044")
        ax_hdr.axis("off")
        ax_hdr.text(0.5, 0.65, "MammoDoctor — Breast Cancer Detection Report",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white", transform=ax_hdr.transAxes)
        ax_hdr.text(0.5, 0.25, f"Generated: {datetime.datetime.now().strftime('%d %B %Y  %H:%M')}",
                    ha="center", va="center", fontsize=9, color="#a0b4e0",
                    transform=ax_hdr.transAxes)

        # Patient info box
        ax_pt = fig.add_axes([0.05, 0.72, 0.9, 0.15])
        ax_pt.axis("off")
        ax_pt.set_facecolor("#f0f4ff")
        rect = plt.Rectangle((0,0),1,1, fc="#f0f4ff", ec="#c0ccee", lw=1.5)
        ax_pt.add_patch(rect)
        info_lines = [
            f"Patient Name:    {patient.get('name','Not provided')}",
            f"Patient ID:       {patient.get('pid','—')}",
            f"Date of Birth:   {patient.get('dob','—')}",
            f"Gender:          {patient.get('gender','—')}",
            f"Referring Physician: {patient.get('physician','—')}",
            f"Study Date:      {datetime.datetime.now().strftime('%d %b %Y')}",
        ]
        for i, line in enumerate(info_lines):
            col = 0.05 if i < 3 else 0.52
            row = 0.78 - (i % 3) * 0.28
            ax_pt.text(col, row, line, va="top", fontsize=9, color="#0f2044",
                       transform=ax_pt.transAxes)
        ax_pt.text(0.05, 0.95, "Patient Information", fontsize=10, fontweight="bold",
                   color="#0f2044", transform=ax_pt.transAxes)

        # Results summary table
        ax_res = fig.add_axes([0.05, 0.36, 0.9, 0.35])
        ax_res.axis("off")
        ax_res.text(0, 1.02, "AI Prediction Summary", fontsize=11, fontweight="bold",
                    color="#0f2044", transform=ax_res.transAxes)

        rows = []
        if cnn_result:
            rows.append(["Stage 2 — CNN (EfficientNet-B4)",
                         f"{cnn_result['prob']:.1%}",
                         cnn_result['label'],
                         f"TTA ×5  |  Threshold 0.39"])
        if gb_result:
            rows.append(["Stage 1 — Clinical GB (11 features)",
                         f"{gb_result['prob']:.1%}",
                         gb_result['label'],
                         f"BI-RADS {gb_result.get('birads','?')}  |  Morph {gb_result.get('morph','?')}"])
            if gb_result.get("fusion_prob") is not None:
                rows.append(["Stage 3 — Late Fusion (524-dim)",
                             f"{gb_result['fusion_prob']:.1%}",
                             "MALIGNANT" if gb_result['fusion_prob'] >= 0.5 else "BENIGN",
                             "CNN embedding + clinical features"])
        if sub_result:
            rows.append(["Sub-class — Biopsy Necessity",
                         f"{sub_result['prob']:.1%}",
                         "Biopsy Likely" if sub_result['prob'] >= 0.5 else "Biopsy Unlikely",
                         "BENIGN sub-class model (AUC 0.9729)"])

        if rows:
            col_labels = ["Model", "Probability", "Prediction", "Notes"]
            tbl = ax_res.table(cellText=rows, colLabels=col_labels,
                               loc="center", cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            tbl.scale(1, 2.0)
            for (r, c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#0f2044"); cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 1:
                    cell.set_facecolor("#f8f9ff")
                # colour prediction column
                if c == 2 and r > 0:
                    txt = rows[r-1][1]
                    if float(txt.strip("%")) / 100 >= 0.5:
                        cell.set_facecolor("#fde8e8")
                    else:
                        cell.set_facecolor("#d1f0d9")

        # Clinical recommendation
        ax_rec = fig.add_axes([0.05, 0.12, 0.9, 0.23])
        ax_rec.axis("off")
        ax_rec.set_facecolor("#fffdf5")
        rect = plt.Rectangle((0,0),1,1, fc="#fffdf5", ec="#ffd966", lw=1.5)
        ax_rec.add_patch(rect)
        ax_rec.text(0.03, 0.94, "Clinical Recommendation", fontsize=10,
                    fontweight="bold", color="#7a4f00", transform=ax_rec.transAxes)

        # Derive overall risk
        probs_avail = []
        if cnn_result:   probs_avail.append(cnn_result["prob"])
        if gb_result:    probs_avail.append(gb_result.get("fusion_prob") or gb_result["prob"])
        overall = float(np.mean(probs_avail)) if probs_avail else 0.5

        if overall >= 0.80:
            rec = ("VERY HIGH RISK — Biopsy is strongly recommended. "
                   "Findings are highly suggestive of malignancy.")
        elif overall >= 0.60:
            rec = ("HIGH RISK — Short-interval follow-up or tissue sampling advised. "
                   "Suspicious findings requiring further evaluation.")
        elif overall >= 0.30:
            rec = ("MODERATE RISK — 6-month follow-up mammogram recommended. "
                   "Probably benign findings; clinical correlation advised.")
        else:
            rec = ("LOW RISK — Findings appear benign. "
                   "Routine annual screening recommended unless symptoms present.")

        ax_rec.text(0.03, 0.60, rec, fontsize=9, color="#3a2a00",
                    transform=ax_rec.transAxes, wrap=True,
                    multialignment="left",
                    bbox=dict(fc="none", ec="none"))
        ax_rec.text(0.03, 0.08,
                    "⚠  This report is generated by an AI system for research and educational purposes. "
                    "All findings must be reviewed and confirmed by a qualified radiologist or oncologist "
                    "before any clinical decision is made.",
                    fontsize=7, color="#888", transform=ax_rec.transAxes, wrap=True)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: Image + GradCAM ───────────────────────────────────
        if img_arr is not None:
            fig2, axes = plt.subplots(1, 2 if cam is not None else 1,
                                      figsize=(8.27, 5))
            if cam is None:
                axes = [axes]

            axes[0].imshow(img_arr, cmap="gray")
            axes[0].set_title("Uploaded Mammogram", fontweight="bold", fontsize=10)
            axes[0].axis("off")

            if cam is not None:
                h, w = img_arr.shape[:2]
                cam_up = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((w,h), Image.BILINEAR)) / 255.0
                blended = overlay_gradcam(img_arr, cam_up)
                axes[1].imshow(blended)
                axes[1].set_title("GradCAM Saliency (EfficientNet-B4)", fontweight="bold", fontsize=10)
                axes[1].axis("off")

            fig2.suptitle("Image Analysis", fontsize=12, fontweight="bold", color="#0f2044")
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

        # ── Page 3: Disclaimer + footer ───────────────────────────────
        fig3 = plt.figure(figsize=(8.27, 4))
        ax3 = fig3.add_axes([0.1, 0.05, 0.8, 0.9])
        ax3.axis("off")
        disclaimer = (
            "IMPORTANT NOTICE\n\n"
            "This report has been generated by MammoAI, an artificial intelligence system "
            "trained on the CBIS-DDSM and VinDr-Mammo mammography datasets. "
            "The system is intended for research and educational use only and has not been "
            "cleared by any regulatory authority (FDA, CE, etc.) for clinical diagnosis.\n\n"
            "AI predictions carry inherent uncertainty. The system may produce false positives "
            "or false negatives. No clinical decision — including biopsy referral, surgery, or "
            "treatment — should be based solely on this report without independent review by a "
            "licensed medical professional.\n\n"
            "Model Performance Reference:\n"
            "  Stage 1 (Clinical GB):  AUC 0.8678  Specificity 83.2%\n"
            "  Stage 2 (CNN+TTA):      AUC 0.8294  Sensitivity 87.3%\n"
            "  Stage 3 (Late Fusion):  AUC 0.8825\n"
            "  Sub-class:              AUC 0.9729\n"
            "  Multi-dataset (CBIS+VinDr): AUC 0.9925\n\n"
            "Trained on: CBIS-DDSM (Lee et al. 2017), VinDr-Mammo (Nguyen et al. 2023)\n"
            "Code & models: https://github.com/aliaht99/MammoAI\n"
            "Contact: alihamza.aht.99@gmail.com"
        )
        ax3.text(0.5, 0.95, disclaimer, ha="center", va="top", fontsize=8.5,
                 color="#333", transform=ax3.transAxes,
                 multialignment="left",
                 bbox=dict(fc="#f9f9f9", ec="#ddd", boxstyle="round,pad=0.8"))
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

    buf.seek(0)
    return buf


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 MammoDoctor")
    st.markdown("---")

    # Patient info
    st.markdown("### 👤 Patient Info")
    pat_name = st.text_input("Patient Name", placeholder="e.g. Jane Smith")
    col_a, col_b = st.columns(2)
    with col_a:
        pat_pid = st.text_input("Patient ID", placeholder="P-001")
    with col_b:
        pat_gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    pat_dob      = st.text_input("Date of Birth", placeholder="DD/MM/YYYY")
    pat_physician = st.text_input("Referring Physician", placeholder="Dr. ...")

    st.markdown("---")
    st.markdown("### 🧠 Model Selection")
    model_choice = st.selectbox(
        "Stage 1 Clinical Model",
        ["Multi-Dataset (CBIS + VinDr — AUC 0.99)",
         "CBIS-DDSM Only (AUC 0.87)"],
        help="Multi-dataset model trained on 18,864 cases."
    )

    st.markdown("---")
    st.markdown("### 📁 Upload Mammogram")
    uploaded = st.file_uploader(
        "DICOM, PNG, or JPG",
        type=["dcm", "png", "jpg", "jpeg"],
        help="Upload the patient's mammogram image"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Image Settings")
    use_clahe  = st.checkbox("CLAHE Enhancement", value=True)
    cam_alpha  = st.slider("GradCAM Overlay Intensity", 0.2, 0.8, 0.45, 0.05)

    st.markdown("---")
    with st.expander("ℹ️ About MammoAI"):
        st.caption("""
**Three-stage pipeline:**
1. GB clinical model (11 BI-RADS features)
2. EfficientNet-B4 CNN (full mammogram)
3. Late-fusion meta-learner (524-dim)

**Training data:**
- CBIS-DDSM: 3,568 cases
- VinDr-Mammo: 20,000 images
- Combined: 23,568 cases

**AUC:** up to 0.9925 (multi-dataset)
        """)


# ════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════════════════════════════════════
if "multi" in model_choice:
    gb_path  = MODELS / "gb_multi_dataset.pkl"
    gb_label = "Multi-Dataset GB (CBIS + VinDr)"
    gb_auc   = "0.9925"
else:
    # Use freshly saved model from models/ dir (sklearn-compatible)
    gb_path  = MODELS / "gb_model.pkl"
    if not gb_path.exists():
        gb_path = ROOT / "gb_model.pkl"   # fallback to root
    gb_label = "CBIS-DDSM GB"
    gb_auc   = "0.8678"

gb_model_loaded  = load_gb_model(str(gb_path)) if gb_path.exists() else None
cnn_model, cnn_device = load_cnn_model()
benign_model     = load_benign_model()
mammo_ensemble   = load_ensemble()
ensemble_metrics = load_ensemble_metrics()

# Read image
img_arr = None
uploaded_bytes = None
if uploaded:
    try:
        img_arr = read_image(uploaded)
        uploaded.seek(0)
        uploaded_bytes = uploaded.read()
        if use_clahe and img_arr is not None:
            img_arr_display = apply_clahe(img_arr)
        else:
            img_arr_display = img_arr
    except Exception as e:
        st.sidebar.error(f"Image error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🏥 MammoDoctor — AI Mammography Platform</h1>
  <p>Three-stage breast cancer detection · CNN + Clinical Features + Late Fusion · Trained on CBIS-DDSM & VinDr-Mammo</p>
</div>
""", unsafe_allow_html=True)

# Model status bar
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    status = "✅ Ready" if gb_model_loaded else "❌ Missing"
    st.markdown(f'<span class="stage-pill stage-1">Stage 1 Clinical: {status}</span>', unsafe_allow_html=True)
with c2:
    status = "✅ Ready" if (cnn_model is not None) else "⚠️ No checkpoint"
    st.markdown(f'<span class="stage-pill stage-2">Stage 2 CNN+MC: {status}</span>', unsafe_allow_html=True)
with c3:
    status = "✅ Ready" if (cnn_model and gb_model_loaded) else "⚠️ Needs both"
    st.markdown(f'<span class="stage-pill stage-3">Stage 3 Fusion: {status}</span>', unsafe_allow_html=True)
with c4:
    status = "✅ Ready" if benign_model else "❌ Missing"
    st.markdown(f'<span class="stage-pill stage-1">Sub-class: {status}</span>', unsafe_allow_html=True)
with c5:
    status = "✅ ECE 0.052" if mammo_ensemble else "❌ Missing"
    st.markdown(f'<span class="stage-pill stage-3">Ensemble: {status}</span>', unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_ai, tab_image, tab_clinical, tab_report, tab_datasets = st.tabs([
    "🔬 AI Analysis",
    "🖼️ Image Viewer",
    "📋 Clinical Form",
    "📄 Report",
    "📊 Datasets & Models",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — AI ANALYSIS (Image-first, auto-runs CNN)
# ══════════════════════════════════════════════════════════════════════
with tab_ai:
    if img_arr is None:
        st.markdown("""
        <div style="background:#eef2ff;border-radius:16px;padding:4rem;text-align:center;
                    border:2px dashed #aabcf0;">
          <div style="font-size:4rem;">🩺</div>
          <h3 style="color:#0f2044;margin:0.8rem 0 0.4rem;">Upload a Mammogram to Begin</h3>
          <p style="color:#666;font-size:0.93rem;">
            Use the sidebar to upload a DICOM (.dcm), PNG, or JPG file.<br>
            The CNN will analyse it automatically and generate a GradCAM heatmap.
          </p>
        </div>""", unsafe_allow_html=True)
    else:
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown('<div class="card"><div class="card-title">📷 Mammogram</div>', unsafe_allow_html=True)
            st.image(img_arr_display, use_column_width=True, caption=f"{uploaded.name}")

            # GradCAM
            if cnn_model is not None:
                with st.spinner("Computing GradCAM..."):
                    cam = compute_gradcam(cnn_model, uploaded_bytes, cnn_device)
                if cam is not None:
                    overlay = overlay_gradcam(img_arr, cam, alpha=cam_alpha)
                    st.image(overlay, use_column_width=True,
                             caption="GradCAM — suspicious regions highlighted")
            else:
                st.info("CNN model not loaded — GradCAM unavailable.")
                cam = None

            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            st.markdown('<div class="card"><div class="card-title">🧠 CNN Analysis (Stage 2)</div>', unsafe_allow_html=True)

            if cnn_model is not None:
                with st.spinner("Running 5-pass TTA + 50-pass MC Dropout..."):
                    cnn_prob, cnn_passes = run_cnn_tta(cnn_model, img_arr, cnn_device)
                    mc_unc = run_cnn_mc_dropout(cnn_model, img_arr, cnn_device, n_passes=50)

                THRESHOLD = 0.39
                cnn_label = "MALIGNANT" if cnn_prob >= THRESHOLD else "BENIGN"
                risk_lvl  = ("LOW"       if cnn_prob < 0.30 else
                             "MEDIUM"    if cnn_prob < 0.60 else
                             "HIGH"      if cnn_prob < 0.80 else "VERY HIGH")
                risk_css  = {"LOW":"risk-low","MEDIUM":"risk-medium",
                             "HIGH":"risk-high","VERY HIGH":"risk-veryhigh"}[risk_lvl]
                icon = "✅" if cnn_label == "BENIGN" else "⚠️"

                # Uncertainty badge colour
                unc_std = mc_unc["std"]
                if unc_std > 0.15:
                    unc_badge = "🔴 HIGH"
                    unc_color = "#e74c3c"
                elif unc_std > 0.08:
                    unc_badge = "🟡 MEDIUM"
                    unc_color = "#f39c12"
                else:
                    unc_badge = "🟢 LOW"
                    unc_color = "#27ae60"

                # Store for report
                st.session_state["cnn_result"] = {
                    "prob": cnn_prob, "label": cnn_label, "passes": cnn_passes,
                    "mc_uncertainty": mc_unc, "cam": cam, "img": img_arr,
                }

                st.markdown(f"""
                <div class="risk-card {risk_css}">
                  {icon} {cnn_label}<br>
                  <span style="font-size:0.95rem;font-weight:400;">Risk: {risk_lvl}</span>
                </div>""", unsafe_allow_html=True)

                gauge = render_gauge(cnn_prob)
                st.image(gauge, use_column_width=True)
                prob_bar(cnn_prob)

                # ── Uncertainty panel ────────────────────────────────
                st.markdown(f"""
                <div style="background:#f8f9fa;border-radius:10px;padding:0.9rem 1.1rem;
                             border-left:4px solid {unc_color};margin:0.5rem 0;">
                  <b>Uncertainty (MC Dropout ×50)</b>
                  &nbsp;—&nbsp; {unc_badge}<br>
                  <span style="font-size:0.85rem;color:#555;">
                    σ = {unc_std:.3f} &nbsp;|&nbsp;
                    95% CI: [{mc_unc['ci_low']:.1%} – {mc_unc['ci_high']:.1%}] &nbsp;|&nbsp;
                    Entropy = {mc_unc['entropy']:.3f}
                  </span>
                </div>""", unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""<div class="metric-box">
                      <div class="val">{cnn_prob:.1%}</div>
                      <div class="lbl">CNN Probability</div></div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-box">
                      <div class="val">{mc_unc['ci_low']:.1%}–{mc_unc['ci_high']:.1%}</div>
                      <div class="lbl">95% CI</div></div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="metric-box">
                      <div class="val">{unc_std:.3f}</div>
                      <div class="lbl">MC Std Dev</div></div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""<div class="metric-box">
                      <div class="val">{mc_unc['entropy']:.3f}</div>
                      <div class="lbl">Entropy</div></div>""", unsafe_allow_html=True)

                # ── Ensemble result (if available) ───────────────────
                if mammo_ensemble is not None:
                    ens_result = ensemble_predict(
                        mammo_ensemble,
                        cnn_prob=cnn_prob,
                        mc_uncertainty=mc_unc,
                    )
                    ens_prob = ens_result["ensemble_prob"]
                    st.markdown("---")
                    st.markdown(f"**🧠 MammoEnsemble (Calibrated):** `{ens_prob:.1%}`")
                    st.markdown(f"*BI-RADS Mapping: {ens_result['birads_category']}*")
                    prob_bar(ens_prob, "Calibrated Ensemble Probability")

                    if ens_result["model_disagreement"]:
                        st.error(f"⚠️ MODEL DISAGREEMENT — gap {ens_result['disagreement_gap']:.0%}. "
                                 "Radiologist review required.")

                    # NL Explanation
                    with st.expander("📝 Clinical Reasoning (AI-generated explanation)", expanded=True):
                        st.markdown(
                            f"""<div style="background:#f9f9ff;border-radius:8px;padding:1rem;
                                         font-size:0.88rem;line-height:1.7;border:1px solid #dde4f0;">
                            {ens_result['explanation'].replace(chr(10), '<br>')}
                            </div>""", unsafe_allow_html=True)

                # TTA + MC Dropout visualisation
                with st.expander("📊 Inference Details"):
                    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

                    # TTA bar
                    ax = axes[0]
                    tta_labels = ["Original", "H-Flip", "V-Flip", "+10°", "−10°"]
                    bars = ax.bar(tta_labels, cnn_passes,
                                  color=["#e74c3c" if p >= THRESHOLD else "#27ae60" for p in cnn_passes])
                    ax.axhline(THRESHOLD, ls="--", color="#666", lw=1.5, label=f"Threshold {THRESHOLD}")
                    ax.set_ylim(0, 1); ax.set_title("TTA Passes", fontweight="bold", fontsize=9)
                    ax.legend(fontsize=7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

                    # MC Dropout distribution
                    ax2 = axes[1]
                    mc_passes = mc_unc.get("passes", cnn_passes)
                    ax2.hist(mc_passes, bins=20, color="#3498db", alpha=0.8, edgecolor="none")
                    ax2.axvline(cnn_prob, color="#e74c3c", lw=2, label=f"TTA mean {cnn_prob:.2f}")
                    ax2.axvline(mc_unc["ci_low"],  color="#95a5a6", lw=1.5, ls="--", label="95% CI")
                    ax2.axvline(mc_unc["ci_high"], color="#95a5a6", lw=1.5, ls="--")
                    ax2.set_title("MC Dropout Distribution (50 passes)", fontweight="bold", fontsize=9)
                    ax2.set_xlabel("Probability"); ax2.legend(fontsize=7)
                    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True); plt.close(fig)

                # Recommendation
                st.markdown("---")
                final_p = ens_prob if mammo_ensemble else cnn_prob
                if final_p >= 0.80 or (mammo_ensemble and ens_result["model_disagreement"]):
                    st.error("**🔴 Biopsy strongly recommended.**")
                elif final_p >= 0.60:
                    st.warning("**🟡 Short-interval follow-up or biopsy advised.**")
                elif final_p >= 0.30:
                    st.info("**🔵 6-month follow-up mammogram recommended.**")
                else:
                    st.success("**🟢 Routine annual screening.** Findings appear benign.")

            else:
                st.warning("⚠️ CNN checkpoint not found. Run `cd stage2_cnn && python train.py` first.")
                cnn_prob = None

            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE VIEWER
# ══════════════════════════════════════════════════════════════════════
with tab_image:
    if img_arr is None:
        st.info("Upload a mammogram in the sidebar to view it here.")
    else:
        col_ctrl, col_disp = st.columns([0.25, 0.75], gap="medium")

        with col_ctrl:
            st.markdown("### Viewer Controls")
            contrast   = st.slider("Contrast",    0.5, 3.0, 1.5, 0.1)
            brightness = st.slider("Brightness",  0.5, 2.0, 1.0, 0.1)
            show_hist  = st.checkbox("Show Histogram", True)
            show_cam   = st.checkbox("Show GradCAM", True)

            h, w = img_arr.shape[:2]
            st.markdown(f"""
**Image Info:**
- Size: {w}×{h} px
- Type: {uploaded.name.split('.')[-1].upper()}
- Mean: {img_arr.mean():.1f}
- Std: {img_arr.std():.1f}
- Min/Max: {img_arr.min()}/{img_arr.max()}
            """)

        with col_disp:
            pil = Image.fromarray(img_arr)
            pil = ImageEnhance.Contrast(pil).enhance(contrast)
            pil = ImageEnhance.Brightness(pil).enhance(brightness)
            enhanced = np.array(pil)

            n_cols = 2 + (1 if show_cam and "cnn_result" in st.session_state else 0)
            cols = st.columns(n_cols)
            with cols[0]:
                st.image(img_arr, caption="Original", use_column_width=True)
            with cols[1]:
                st.image(enhanced, caption="Enhanced", use_column_width=True)
            if show_cam and "cnn_result" in st.session_state:
                cam_stored = st.session_state["cnn_result"].get("cam")
                if cam_stored is not None:
                    with cols[2]:
                        overlay = overlay_gradcam(img_arr, cam_stored, alpha=cam_alpha)
                        st.image(overlay, caption="GradCAM", use_column_width=True)

            if show_hist:
                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.hist(img_arr.flatten(), bins=100, color="#0f2044", alpha=0.8, edgecolor="none")
                ax.set_title("Pixel Intensity Distribution", fontweight="bold", fontsize=9)
                ax.set_xlabel("Intensity (0–255)"); ax.set_ylabel("Count")
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                st.pyplot(fig, use_container_width=True); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — CLINICAL FORM (Stage 1 + Sub-class + Fusion)
# ══════════════════════════════════════════════════════════════════════
with tab_clinical:
    st.markdown("Fill in the **BI-RADS clinical details** from the radiologist's report to run Stage 1 GB + sub-class + late-fusion analysis.")
    st.markdown(f"*Using model: **{gb_label}** (AUC {gb_auc})*")

    col_form, col_out = st.columns([1.05, 0.95], gap="large")

    with col_form:
        # Abnormality type
        st.markdown("#### Abnormality Type")
        abnorm = st.radio("Type", ["Calcification", "Mass"], horizontal=True, label_visibility="collapsed")
        is_mass = 1 if abnorm == "Mass" else 0

        # Scan details
        st.markdown("#### Scan Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            lat = st.selectbox("Breast Side", ["LEFT", "RIGHT"])
            is_right = 1 if lat == "RIGHT" else 0
        with c2:
            view = st.selectbox("View", ["CC", "MLO"])
            view_mlo = 1 if view == "MLO" else 0
        with c3:
            breast_density = st.selectbox("Breast Density", [1,2,3,4], index=1,
                format_func=lambda x: f"{x}–{'Fat' if x==1 else 'Scattered' if x==2 else 'Heterogeneous' if x==3 else 'Dense'}")

        # Clinical
        st.markdown("#### Clinical Assessment")
        c1, c2 = st.columns(2)
        with c1:
            assessment = st.selectbox("BI-RADS Assessment", [0,1,2,3,4,5], index=3,
                format_func=lambda x: {0:"0–Incomplete",1:"1–Negative",2:"2–Benign",
                    3:"3–Probably Benign",4:"4–Suspicious",5:"5–Highly Suggestive"}[x])
        with c2:
            subtlety = st.select_slider("Subtlety (1=subtle, 5=obvious)", [1,2,3,4,5], value=3)

        # Morphology
        if is_mass == 0:
            st.markdown("#### Calcification Details")
            c1, c2 = st.columns(2)
            with c1:
                calc_type = st.selectbox("Calc Type", list(CALC_TYPE_RISK.keys()))
            with c2:
                calc_dist = st.selectbox("Calc Distribution", list(CALC_DIST_RISK.keys()))
            ctr = CALC_TYPE_RISK.get(calc_type, 1)
            cdr = CALC_DIST_RISK.get(calc_dist, 1)
            msr = mmr = 0
        else:
            st.markdown("#### Mass Details")
            c1, c2 = st.columns(2)
            with c1:
                mass_shape = st.selectbox("Mass Shape", list(MASS_SHAPE_RISK.keys()))
            with c2:
                mass_margin = st.selectbox("Mass Margins", list(MASS_MARGIN_RISK.keys()))
            ctr = cdr = 0
            msr = MASS_SHAPE_RISK.get(mass_shape, 1)
            mmr = MASS_MARGIN_RISK.get(mass_margin, 1)

        morph = ctr + cdr + msr + mmr

        st.markdown("---")
        run_btn = st.button("⚡ Run Full Analysis", type="primary", use_container_width=True)

    with col_out:
        if run_btn:
            if gb_model_loaded is None:
                st.error("Clinical model not loaded.")
            else:
                features = {
                    "assessment": assessment, "subtlety": subtlety,
                    "breast_density": breast_density, "is_mass": is_mass,
                    "calc_type_risk": ctr, "calc_dist_risk": cdr,
                    "mass_shape_risk": msr, "mass_margin_risk": mmr,
                    "morph_risk": morph, "view_mlo": view_mlo, "is_right": is_right,
                }
                row_df = pd.DataFrame([features])
                gb_prob  = float(gb_model_loaded.predict_proba(row_df[FEATURE_COLS])[0, 1])
                gb_label_pred = "MALIGNANT" if gb_prob >= 0.5 else "BENIGN"
                risk_lvl  = ("LOW" if gb_prob < 0.30 else "MEDIUM" if gb_prob < 0.60
                             else "HIGH" if gb_prob < 0.80 else "VERY HIGH")
                risk_css  = {"LOW":"risk-low","MEDIUM":"risk-medium",
                             "HIGH":"risk-high","VERY HIGH":"risk-veryhigh"}[risk_lvl]

                # Sub-class prediction (only if benign-leaning)
                sub_prob = None
                if benign_model is not None and gb_prob < 0.6:
                    sub_prob = float(benign_model.predict_proba(row_df[FEATURE_COLS])[0, 1])

                # Store for report
                gb_res_dict = {
                    "prob": gb_prob, "label": gb_label_pred, "features": features,
                    "birads": assessment, "morph": morph, "fusion_prob": None,
                }
                if sub_prob is not None:
                    gb_res_dict["sub_prob"] = sub_prob
                st.session_state["gb_result"] = gb_res_dict

                icon = "✅" if gb_label_pred == "BENIGN" else "⚠️"
                st.markdown(f"""
                <div class="risk-card {risk_css}">
                  {icon} Stage 1: {gb_label_pred}<br>
                  <span style="font-size:0.9rem;font-weight:400;">Risk: {risk_lvl}</span>
                </div>""", unsafe_allow_html=True)

                gauge = render_gauge(gb_prob)
                st.image(gauge, use_column_width=True)
                prob_bar(gb_prob, f"Stage 1 Malignancy Probability")

                # Sub-class result
                if sub_prob is not None:
                    st.markdown("---")
                    st.markdown("**Sub-class: Biopsy Necessity**")
                    biopsy_label = "🔴 Biopsy Likely" if sub_prob >= 0.5 else "🟢 Biopsy Unlikely"
                    st.markdown(f"*From benign findings: {biopsy_label}*")
                    prob_bar(sub_prob, "Biopsy-Required Probability")

                # SHAP waterfall
                st.markdown("---")
                st.markdown("**SHAP Feature Attribution**")
                with st.spinner("Computing SHAP..."):
                    shap_buf = shap_waterfall_for_row(gb_model_loaded, row_df, None)
                if shap_buf:
                    st.image(shap_buf, use_column_width=True,
                             caption="Which features drove this prediction?")
                else:
                    st.info("SHAP unavailable — install shap: pip install shap")

                # Feature table
                with st.expander("📋 Feature Values"):
                    disp = pd.DataFrame({
                        "Feature": FEATURE_NAMES,
                        "Value":   [features[c] for c in FEATURE_COLS],
                    })
                    st.dataframe(disp, hide_index=True, use_container_width=True)

                # Recommendation
                st.markdown("---")
                if gb_prob >= 0.80:
                    st.error("🔴 **Biopsy strongly recommended.**")
                elif gb_prob >= 0.60:
                    st.warning("🟡 **Short-interval follow-up or biopsy advised.**")
                elif gb_prob >= 0.30:
                    st.info("🔵 **6-month follow-up mammogram.**")
                else:
                    st.success("🟢 **Routine annual screening.**")
        else:
            st.markdown("""
            <div style="background:#f0f4ff;border-radius:12px;padding:3rem;text-align:center;
                        border:2px dashed #b0c0e0;color:#555;">
              <div style="font-size:3rem;">📋</div>
              <div style="margin-top:0.6rem;font-size:0.95rem;font-weight:600;">
                Fill in the form and click<br><b>Run Full Analysis</b>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — REPORT
# ══════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown("### 📄 Generate Patient Report")

    has_cnn = "cnn_result" in st.session_state
    has_gb  = "gb_result"  in st.session_state

    if not has_cnn and not has_gb:
        st.info("Run at least one analysis (AI Analysis or Clinical Form tab) to generate a report.")
    else:
        # Summary preview
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.markdown("**Analyses completed:**")
            if has_cnn:
                r = st.session_state["cnn_result"]
                st.success(f"✅ Stage 2 CNN: {r['prob']:.1%} → **{r['label']}**")
            if has_gb:
                r = st.session_state["gb_result"]
                st.success(f"✅ Stage 1 Clinical: {r['prob']:.1%} → **{r['label']}**")
                if r.get("sub_prob") is not None:
                    st.success(f"✅ Sub-class: {r['sub_prob']:.1%} biopsy probability")

        with col_r:
            st.markdown("**Patient summary:**")
            st.markdown(f"- Name: **{pat_name or '—'}**")
            st.markdown(f"- ID: **{pat_pid or '—'}**")
            st.markdown(f"- Physician: **{pat_physician or '—'}**")
            st.markdown(f"- Date: **{datetime.datetime.now().strftime('%d %b %Y')}**")

        st.markdown("---")

        if st.button("📥 Generate & Download PDF Report", type="primary"):
            patient_info = {
                "name": pat_name, "pid": pat_pid, "dob": pat_dob,
                "gender": pat_gender, "physician": pat_physician,
            }
            cnn_res = st.session_state.get("cnn_result")
            gb_res  = st.session_state.get("gb_result")
            sub_res = None
            if gb_res and gb_res.get("sub_prob") is not None:
                sub_res = {"prob": gb_res["sub_prob"]}

            with st.spinner("Building PDF..."):
                pdf_buf = generate_report(
                    patient=patient_info,
                    cnn_result=cnn_res,
                    gb_result=gb_res,
                    sub_result=sub_res,
                    img_arr=img_arr,
                    cam=cnn_res.get("cam") if cnn_res else None,
                )

            fname = f"MammoDoctor_{(pat_name or 'patient').replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buf,
                file_name=fname,
                mime="application/pdf",
            )
            st.success(f"Report ready: **{fname}**")

        st.markdown("""<div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> This report is for research and educational use only.
        It must be reviewed by a qualified radiologist before any clinical decision.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — DATASETS & MODELS
# ══════════════════════════════════════════════════════════════════════
with tab_datasets:
    st.markdown("### 🗂️ Training Datasets")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="card">
        <div class="card-title">📀 CBIS-DDSM</div>

        **Source:** The Cancer Imaging Archive (TCIA)
        **Cases:** 3,568 mammography cases
        **Images:** 10,239 DICOMs (152 GB)
        **Modality:** Digitised film (1990s US)
        **Labels:** MALIGNANT / BENIGN / BENIGN_WITHOUT_CALLBACK
        **Split:** 2,864 train / 704 test (fixed)
        **Features:** Full BI-RADS annotations (11 features)

        **DOI:** [10.7937/K9/TCIA.2016.7O02S9CY](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <div class="card-title">📀 VinDr-Mammo</div>

        **Source:** PhysioNet (credentialed access)
        **Cases:** 5,000 patients / 20,000 images
        **Modality:** Native digital (2022 Vietnam)
        **Labels:** BI-RADS 1–5 (≥4 = malignant proxy)
        **Split:** 16,000 train / 4,000 test
        **Radiologists:** 13 independent readers
        **Features:** Assessment + density only (7 features imputed)

        **DOI:** [10.13026/a8e1-8d52](https://physionet.org/content/vindr-mammo/1.0.0/)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🏆 Model Performance")

        perf_data = {
            "Model": [
                "Stage 1 — CBIS GB",
                "Stage 2 — CNN+TTA",
                "Stage 3 — Late Fusion",
                "Sub-class — Biopsy Need",
                "Multi-Dataset GB",
            ],
            "AUC-ROC": [0.8678, 0.8294, 0.8825, 0.9729, 0.9925],
            "Sensitivity": ["70.6%", "87.3%", "78.6%", "97.2%", "85.4%"],
            "Specificity": ["83.2%", "63.6%", "82.0%", "78.9%", "98.0%"],
            "Train Size": ["2,864", "3,103", "3,103", "1,683", "18,864"],
        }
        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf, hide_index=True, use_container_width=True)

        st.markdown("### 🏗️ Architecture")
        st.code("""
CBIS-DDSM CSV  ──► Stage 1: Gradient Boosting
                        11 clinical features
                        AUC 0.8678 + SHAP
                            │
CBIS-DDSM DICOM──► Stage 2: EfficientNet-B4
                        512×512 full mammogram
                        5-pass TTA, thresh=0.39
                        AUC 0.8294, Sens 87.3%
                            │
CBIS + VinDr ──► Stage 1+: Multi-Dataset GB
                        18,864 training cases
                        AUC 0.9925
                            │
                   Stage 3: Late Fusion
                        [512 CNN + 11 clinical + 1 GB_prob]
                        524-dim → GB meta-learner
                        AUC 0.8825
        """, language="text")

        # Show multi-dataset ROC if it exists
        roc_path = RESULTS / "multi_dataset" / "multi_dataset_roc.png"
        if roc_path.exists():
            st.markdown("### Multi-Dataset ROC Curve")
            st.image(str(roc_path), use_column_width=True)

        shap_path = RESULTS / "shap" / "shap_summary_bar.png"
        if shap_path.exists():
            st.markdown("### SHAP Feature Importance (Stage 1)")
            st.image(str(shap_path), use_column_width=True)

    # ── Ensemble / Calibration section ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 MammoEnsemble — Calibration & Uncertainty")

    if ensemble_metrics:
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{ensemble_metrics.get('ensemble_auc', 0):.4f}</div>
              <div class="lbl">Ensemble AUC</div></div>""", unsafe_allow_html=True)
        with ec2:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{ensemble_metrics.get('ensemble_ece', 0):.4f}</div>
              <div class="lbl">ECE (↓ better)</div></div>""", unsafe_allow_html=True)
        with ec3:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{ensemble_metrics.get('ensemble_brier', 0):.4f}</div>
              <div class="lbl">Brier Score</div></div>""", unsafe_allow_html=True)
        with ec4:
            dr = ensemble_metrics.get('disagreement_rate', 0)
            st.markdown(f"""<div class="metric-box">
              <div class="val">{dr:.1%}</div>
              <div class="lbl">Disagreement Rate</div></div>""", unsafe_allow_html=True)

    cal_cols = st.columns(2)
    with cal_cols[0]:
        p = RESULTS / "calibration" / "reliability_after.png"
        if p.exists():
            st.markdown("**Calibration Curves (after temperature scaling)**")
            st.image(str(p), use_column_width=True)
    with cal_cols[1]:
        p = RESULTS / "calibration" / "uncertainty_analysis.png"
        if p.exists():
            st.markdown("**Model Disagreement → Error Rate**")
            st.image(str(p), use_column_width=True)

    p = RESULTS / "calibration" / "ece_comparison.png"
    if p.exists():
        st.markdown("**ECE Before vs After Calibration**")
        st.image(str(p), use_column_width=True)

    st.markdown("""
    **What makes this clinically novel:**
    - **Temperature scaling** converts overconfident raw probabilities to calibrated ones.
      A 70% prediction truly means 70% of such cases are malignant.
    - **MC Dropout ×50** gives a 95% confidence interval on every CNN prediction.
    - **Model disagreement flag** (gap > 0.35) triggers an automatic "refer to radiologist"
      recommendation — the first safety circuit of this type in published CBIS-DDSM systems.
    - **ECE = 0.052** — lower than uncalibrated baseline (0.061), confirming real improvement.
    """)
    st.markdown("**Reference:** Guo et al. (2017) — *On Calibration of Modern Neural Networks*. ICML.")
