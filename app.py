"""
Breast Cancer Detection – Streamlit Web App
Upload any mammogram image (DICOM / PNG / JPG) and fill in clinical details
to get an AI-powered malignancy risk assessment.
"""

import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageEnhance, ImageFilter

warnings.filterwarnings("ignore")

# ── optional DICOM support ──────────────────────────────────────────────────
try:
    import pydicom
    DICOM_OK = True
except ImportError:
    DICOM_OK = False

# ── optional skimage for texture metrics ───────────────────────────────────
try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage import exposure
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

# ──────────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path(__file__).parent / "gb_model.pkl"
RESULTS_DIR  = Path(__file__).parent / "results"

FEATURE_COLS = [
    "assessment", "subtlety", "breast_density", "is_mass",
    "calc_type_risk", "calc_dist_risk", "mass_shape_risk", "mass_margin_risk",
    "morph_risk", "view_mlo", "is_right",
]
FEATURE_LABELS = {
    "assessment":       "BI-RADS Assessment",
    "subtlety":         "Subtlety",
    "breast_density":   "Breast Density",
    "is_mass":          "Abnormality Type (Mass=1)",
    "calc_type_risk":   "Calc Type Risk",
    "calc_dist_risk":   "Calc Distribution Risk",
    "mass_shape_risk":  "Mass Shape Risk",
    "mass_margin_risk": "Mass Margin Risk",
    "morph_risk":       "Morphology Risk (combined)",
    "view_mlo":         "View MLO (1) / CC (0)",
    "is_right":         "Right Breast (1) / Left (0)",
}

# ─── page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MammoAI – Breast Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; letter-spacing: -0.5px; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.75; font-size: 0.95rem; }

    .risk-card {
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        text-align: center;
        font-weight: 700;
        font-size: 1.5rem;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .risk-low      { background:#d4edda; color:#155724; border:2px solid #c3e6cb; }
    .risk-medium   { background:#fff3cd; color:#856404; border:2px solid #ffeeba; }
    .risk-high     { background:#f8d7da; color:#721c24; border:2px solid #f5c6cb; }
    .risk-veryhigh { background:#721c24; color:#ffffff; border:2px solid #f5c6cb; }

    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-box .val { font-size: 2rem; font-weight: 800; color: #0f3460; }
    .metric-box .lbl { font-size: 0.78rem; color: #666; margin-top: 2px; }

    .disclaimer {
        background: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        font-size: 0.83rem;
        color: #555;
        margin-top: 1.5rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f3460;
        margin: 1.2rem 0 0.5rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.3rem;
    }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── helpers ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_image(uploaded_file) -> np.ndarray | None:
    """Return a numpy uint8 array from DICOM, PNG, or JPG."""
    name = uploaded_file.name.lower()
    if name.endswith(".dcm"):
        if not DICOM_OK:
            st.error("pydicom not installed. Run: pip install pydicom")
            return None
        ds = pydicom.dcmread(uploaded_file)
        arr = ds.pixel_array.astype(float)
        arr = ((arr - arr.min()) / (arr.ptp() + 1e-8) * 255).astype(np.uint8)
        return arr
    else:
        pil = Image.open(uploaded_file).convert("L")  # grayscale
        return np.array(pil)


def enhance_image(arr: np.ndarray, contrast: float, brightness: float,
                  clahe: bool, denoise: bool) -> np.ndarray:
    pil = Image.fromarray(arr)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    if denoise:
        pil = pil.filter(ImageFilter.MedianFilter(size=3))
    if clahe and SKIMAGE_OK:
        from skimage import exposure as skexp
        img_eq = skexp.equalize_adapthist(np.array(pil), clip_limit=0.03)
        pil = Image.fromarray((img_eq * 255).astype(np.uint8))
    return np.array(pil)


def extract_image_metrics(arr: np.ndarray) -> dict:
    flat = arr.flatten().astype(float)
    metrics = {
        "Mean Intensity":      round(float(np.mean(flat)), 2),
        "Std Dev":             round(float(np.std(flat)), 2),
        "Contrast (p95-p5)":   round(float(np.percentile(flat, 95) - np.percentile(flat, 5)), 2),
        "Brightness (p75)":    round(float(np.percentile(flat, 75)), 2),
    }
    if SKIMAGE_OK:
        gray = (arr / 255.0 * 63).astype(np.uint8).clip(0, 63)
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2],
                            levels=64, symmetric=True, normed=True)
        metrics["GLCM Contrast"]     = round(float(graycoprops(glcm, "contrast").mean()), 4)
        metrics["GLCM Energy"]       = round(float(graycoprops(glcm, "energy").mean()), 4)
        metrics["GLCM Homogeneity"]  = round(float(graycoprops(glcm, "homogeneity").mean()), 4)
        metrics["GLCM Correlation"]  = round(float(graycoprops(glcm, "correlation").mean()), 4)
    return metrics


def render_gauge(prob: float):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    theta_start, theta_end = np.pi, 0   # 180° → 0°
    n = 200
    thetas = np.linspace(theta_start, theta_end, n)

    cmap = cm.get_cmap("RdYlGn_r")
    for i in range(n - 1):
        t = i / (n - 1)
        ax.plot(thetas[i:i+2], [0.85, 0.85], color=cmap(t), lw=8, solid_capstyle="round")

    needle = theta_start + (theta_end - theta_start) * prob
    ax.annotate("", xy=(needle, 0.75), xytext=(needle, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="#1a1a2e", lw=2.5, mutation_scale=18))

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.text(0, -0.15, f"{prob:.1%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#1a1a2e",
            transform=ax.transData)
    ax.text(np.pi * 1.05, 0.95, "0%", ha="center", fontsize=7, color="#555")
    ax.text(0,            0.95, "100%", ha="center", fontsize=7, color="#555")
    ax.text(np.pi / 2,   1.05, "50%", ha="center", fontsize=7, color="#555")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def probability_bar(prob: float):
    col = ("#28a745" if prob < 0.30 else
           "#ffc107" if prob < 0.60 else
           "#dc3545" if prob < 0.80 else "#6f1023")
    bar_pct = int(prob * 100)
    st.markdown(f"""
    <div style="background:#e9ecef;border-radius:8px;height:26px;overflow:hidden;margin:6px 0 14px;">
      <div style="width:{bar_pct}%;background:{col};height:100%;border-radius:8px;
                  display:flex;align-items:center;justify-content:center;
                  color:white;font-weight:700;font-size:0.85rem;transition:width 0.5s;">
        {bar_pct}%
      </div>
    </div>""", unsafe_allow_html=True)


def feature_importance_chart(model):
    clf = model.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        return None
    imp = clf.feature_importances_
    labels = [FEATURE_LABELS[c] for c in FEATURE_COLS]
    idx = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#0f3460" if imp[i] > np.median(imp) else "#90b4d8" for i in idx]
    bars = ax.barh([labels[i] for i in idx], imp[idx], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title("Feature Importance (Gradient Boosting)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def histogram_chart(arr: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.hist(arr.flatten(), bins=80, color="#0f3460", alpha=0.8, edgecolor="none")
    ax.set_title("Pixel Intensity Distribution", fontsize=9, fontweight="bold")
    ax.set_xlabel("Intensity (0-255)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/breast-cancer.png", width=72)
    st.markdown("## MammoAI Settings")
    st.markdown("---")

    st.markdown("### Upload Mammogram")
    uploaded = st.file_uploader(
        "Accepts DICOM (.dcm), PNG, JPG",
        type=["dcm", "png", "jpg", "jpeg"],
        help="Upload any mammogram image. DICOM files are read natively."
    )

    img_arr = None
    if uploaded:
        img_arr = load_image(uploaded)
        if img_arr is not None:
            st.success(f"Loaded: {uploaded.name}\nSize: {img_arr.shape[1]}×{img_arr.shape[0]} px")

    st.markdown("---")
    st.markdown("### Image Enhancement")
    contrast   = st.slider("Contrast",   0.5, 3.0, 1.5, 0.1)
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    clahe_on   = st.checkbox("CLAHE (adaptive histogram eq.)", value=True)
    denoise_on = st.checkbox("Median denoise", value=False)

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "Trained on **CBIS-DDSM** (2,864 mammography cases).  \n"
        "Model: Gradient Boosting | AUC-ROC: 0.87"
    )


# ─── main content ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🩺 MammoAI — Breast Cancer Detection</h1>
  <p>AI-powered mammogram analysis · CBIS-DDSM dataset · Gradient Boosting model</p>
</div>
""", unsafe_allow_html=True)

model = load_model()

tab_predict, tab_image, tab_info = st.tabs([
    "🔬 Predict", "🖼️ Image Viewer", "📊 Model Info"
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("Fill in the **clinical / radiological details** from the mammogram report, then click **Run Analysis**.")

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        # ── Abnormality type ──────────────────────────────────────
        st.markdown('<div class="section-title">Abnormality Type</div>', unsafe_allow_html=True)
        abnorm_type = st.radio(
            "Type",
            ["Calcification", "Mass"],
            horizontal=True,
            label_visibility="collapsed",
        )
        is_mass = 1 if abnorm_type == "Mass" else 0

        # ── Image / scan info ─────────────────────────────────────
        st.markdown('<div class="section-title">Scan Details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            laterality = st.selectbox("Breast Side", ["LEFT", "RIGHT"])
            is_right   = 1 if laterality == "RIGHT" else 0
        with c2:
            view = st.selectbox("View", ["CC", "MLO"])
            view_mlo = 1 if view == "MLO" else 0
        with c3:
            breast_density = st.selectbox(
                "Breast Density",
                [1, 2, 3, 4],
                index=1,
                format_func=lambda x: f"{x} – {'Almost fat' if x==1 else 'Scattered' if x==2 else 'Heterogeneous' if x==3 else 'Extremely dense'}",
            )

        # ── Clinical assessment ───────────────────────────────────
        st.markdown('<div class="section-title">Clinical Assessment</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            assessment = st.selectbox(
                "BI-RADS Assessment",
                [0, 1, 2, 3, 4, 5],
                index=3,
                format_func=lambda x: {
                    0: "0 – Incomplete",
                    1: "1 – Negative",
                    2: "2 – Benign",
                    3: "3 – Probably Benign",
                    4: "4 – Suspicious",
                    5: "5 – Highly Suggestive",
                }[x],
                help="BI-RADS score assigned by the radiologist"
            )
        with c2:
            subtlety = st.select_slider(
                "Subtlety",
                options=[1, 2, 3, 4, 5],
                value=3,
                help="1=barely visible, 5=obvious"
            )

        # ── Calcification details (hidden if mass) ────────────────
        if is_mass == 0:
            st.markdown('<div class="section-title">Calcification Details</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                calc_type = st.selectbox("Calc Type", [
                    "PLEOMORPHIC", "AMORPHOUS", "HETEROGENEOUS",
                    "FINE_LINEAR_BRANCHING", "PUNCTATE", "ROUND_AND_REGULAR",
                    "COARSE", "MILK_OF_CALCIUM", "N/A",
                ])
            with c2:
                calc_dist = st.selectbox("Calc Distribution", [
                    "CLUSTERED", "LINEAR", "SEGMENTAL", "REGIONAL", "DIFFUSELY_SCATTERED",
                ])

            CALC_TYPE_RISK = {
                "PLEOMORPHIC": 3, "AMORPHOUS": 2, "HETEROGENEOUS": 2,
                "FINE_LINEAR_BRANCHING": 3, "PUNCTATE": 1, "ROUND_AND_REGULAR": 0,
                "COARSE": 0, "MILK_OF_CALCIUM": 0, "N/A": 1,
            }
            CALC_DIST_RISK = {
                "CLUSTERED": 2, "LINEAR": 3, "SEGMENTAL": 2,
                "REGIONAL": 1, "DIFFUSELY_SCATTERED": 0,
            }
            calc_type_risk  = CALC_TYPE_RISK.get(calc_type, 1)
            calc_dist_risk  = CALC_DIST_RISK.get(calc_dist, 1)
            mass_shape_risk = 0
            mass_margin_risk= 0
        else:
            # ── Mass details ──────────────────────────────────────
            st.markdown('<div class="section-title">Mass Details</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                mass_shape = st.selectbox("Mass Shape", [
                    "IRREGULAR", "IRREGULAR-ARCHITECTURAL_DISTORTION",
                    "LOBULATED", "OVAL", "ROUND", "ARCHITECTURAL_DISTORTION",
                ])
            with c2:
                mass_margin = st.selectbox("Mass Margins", [
                    "SPICULATED", "ILL_DEFINED", "OBSCURED",
                    "MICROLOBULATED", "CIRCUMSCRIBED",
                ])

            MASS_SHAPE_RISK = {
                "IRREGULAR": 3, "IRREGULAR-ARCHITECTURAL_DISTORTION": 3,
                "LOBULATED": 2, "OVAL": 1, "ROUND": 1,
                "ARCHITECTURAL_DISTORTION": 2,
            }
            MASS_MARGIN_RISK = {
                "SPICULATED": 3, "ILL_DEFINED": 2, "OBSCURED": 1,
                "MICROLOBULATED": 2, "CIRCUMSCRIBED": 0,
            }
            calc_type_risk  = 0
            calc_dist_risk  = 0
            mass_shape_risk  = MASS_SHAPE_RISK.get(mass_shape, 1)
            mass_margin_risk = MASS_MARGIN_RISK.get(mass_margin, 1)

        morph_risk = calc_type_risk + calc_dist_risk + mass_shape_risk + mass_margin_risk

        st.markdown("---")
        run = st.button("🔬 Run Analysis", type="primary", use_container_width=True)

    # ── results panel ─────────────────────────────────────────────────────
    with col_result:
        if run:
            features = {
                "assessment":       assessment,
                "subtlety":         subtlety,
                "breast_density":   breast_density,
                "is_mass":          is_mass,
                "calc_type_risk":   calc_type_risk,
                "calc_dist_risk":   calc_dist_risk,
                "mass_shape_risk":  mass_shape_risk,
                "mass_margin_risk": mass_margin_risk,
                "morph_risk":       morph_risk,
                "view_mlo":         view_mlo,
                "is_right":         is_right,
            }
            row  = pd.DataFrame([features])[FEATURE_COLS]
            prob = float(model.predict_proba(row)[0, 1])
            pred = "MALIGNANT" if prob >= 0.5 else "BENIGN"

            risk_level = ("LOW"       if prob < 0.30 else
                          "MEDIUM"    if prob < 0.60 else
                          "HIGH"      if prob < 0.80 else
                          "VERY HIGH")
            risk_css   = {"LOW": "risk-low", "MEDIUM": "risk-medium",
                          "HIGH": "risk-high", "VERY HIGH": "risk-veryhigh"}[risk_level]

            # Risk card
            icon = "✅" if pred == "BENIGN" else "⚠️"
            st.markdown(f"""
            <div class="risk-card {risk_css}">
              {icon} {pred}<br>
              <span style="font-size:1rem;font-weight:400;">Risk Level: {risk_level}</span>
            </div>""", unsafe_allow_html=True)

            # gauge
            gauge_buf = render_gauge(prob)
            st.image(gauge_buf, caption="Malignancy Probability", use_column_width=True)
            probability_bar(prob)

            # metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class="metric-box">
                  <div class="val">{prob:.1%}</div>
                  <div class="lbl">Malignancy Prob.</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-box">
                  <div class="val">{morph_risk}</div>
                  <div class="lbl">Morphology Risk</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-box">
                  <div class="val">{assessment}</div>
                  <div class="lbl">BI-RADS Score</div>
                </div>""", unsafe_allow_html=True)

            # feature input summary
            with st.expander("📋 Feature Summary", expanded=False):
                df_show = pd.DataFrame({
                    "Feature":  [FEATURE_LABELS[c] for c in FEATURE_COLS],
                    "Value":    [features[c] for c in FEATURE_COLS],
                })
                st.dataframe(df_show, hide_index=True, use_container_width=True)

            # recommendations
            st.markdown("---")
            st.markdown("#### 📌 Clinical Notes")
            if prob >= 0.80:
                st.error("**Biopsy strongly recommended.** High suspicion of malignancy.")
            elif prob >= 0.60:
                st.warning("**Short-interval follow-up or biopsy** advised. Suspicious findings.")
            elif prob >= 0.30:
                st.info("**6-month follow-up mammogram** recommended. Probably benign.")
            else:
                st.success("**Routine screening** (annually). Findings appear benign.")

            st.markdown("""<div class="disclaimer">
            ⚠️ <b>Disclaimer:</b> This tool is for <b>educational and research purposes only</b>.
            It does not constitute medical advice. Always consult a qualified radiologist or
            oncologist for clinical decisions.
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#f0f4ff;border-radius:12px;padding:2.5rem;text-align:center;
                        color:#555;border:2px dashed #b0c0e0;margin-top:1rem;">
              <div style="font-size:3rem;">🔬</div>
              <div style="font-size:1rem;font-weight:600;margin-top:0.5rem;">
                Fill in the clinical details and click<br><b>Run Analysis</b>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — IMAGE VIEWER
# ══════════════════════════════════════════════════════════════════
with tab_image:
    if img_arr is None:
        st.markdown("""
        <div style="background:#f0f4ff;border-radius:12px;padding:3rem;text-align:center;
                    color:#555;border:2px dashed #b0c0e0;">
          <div style="font-size:3.5rem;">🖼️</div>
          <div style="font-size:1rem;font-weight:600;margin-top:0.8rem;">
            Upload a mammogram in the <b>sidebar</b> to view it here.<br>
            Supports: DICOM (.dcm), PNG, JPG
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        enhanced = enhance_image(img_arr, contrast, brightness, clahe_on, denoise_on)

        col_orig, col_enh = st.columns(2, gap="medium")
        with col_orig:
            st.markdown("#### Original Image")
            st.image(img_arr, caption="Original", use_column_width=True, clamp=True)

        with col_enh:
            st.markdown("#### Enhanced Image")
            st.image(enhanced, caption="Enhanced (adjust sliders in sidebar)",
                     use_column_width=True, clamp=True)

        st.markdown("---")
        st.markdown("#### 📊 Image Statistics")
        metrics = extract_image_metrics(img_arr)

        n = len(metrics)
        cols = st.columns(min(n, 4))
        for i, (k, v) in enumerate(metrics.items()):
            with cols[i % 4]:
                st.metric(k, v)

        col_hist, col_info = st.columns([1.2, 0.8], gap="medium")
        with col_hist:
            hist_buf = histogram_chart(img_arr)
            st.image(hist_buf, use_column_width=True)

        with col_info:
            st.markdown("#### Image Info")
            h, w = img_arr.shape[:2]
            st.markdown(f"""
| Property | Value |
|---|---|
| Width | {w} px |
| Height | {h} px |
| Channels | {'Grayscale' if img_arr.ndim==2 else img_arr.shape[2]} |
| Min intensity | {int(img_arr.min())} |
| Max intensity | {int(img_arr.max())} |
| File type | {uploaded.name.split('.')[-1].upper()} |
""")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ══════════════════════════════════════════════════════════════════
with tab_info:
    col_a, col_b = st.columns([1.1, 0.9], gap="large")

    with col_a:
        st.markdown("### Model Details")
        st.markdown("""
| Property | Value |
|---|---|
| **Algorithm** | Gradient Boosting Classifier |
| **Dataset** | CBIS-DDSM (Curated Breast Imaging Subset of DDSM) |
| **Training samples** | 2,864 mammography cases |
| **Test AUC-ROC** | 0.8678 |
| **Sensitivity** | 0.7065 (recall for Malignant) |
| **Specificity** | 0.8318 (recall for Benign) |
| **Avg Precision** | 0.8071 |
| **CV AUC (5-fold)** | 0.8647 |
""")

        st.markdown("### Model Comparison")
        comparison = pd.DataFrame([
            {"Model": "Gradient Boosting", "AUC-ROC": 0.8678, "Sensitivity": 0.71, "Specificity": 0.83},
            {"Model": "Random Forest",     "AUC-ROC": 0.8457, "Sensitivity": 0.79, "Specificity": 0.71},
            {"Model": "SVM (RBF)",         "AUC-ROC": 0.8410, "Sensitivity": 0.85, "Specificity": 0.63},
            {"Model": "Logistic Reg.",     "AUC-ROC": 0.7930, "Sensitivity": 0.79, "Specificity": 0.57},
        ])
        st.dataframe(comparison, hide_index=True, use_container_width=True)

        st.markdown("### Feature Descriptions")
        st.markdown("""
| Feature | Description |
|---|---|
| **BI-RADS Assessment** | Radiologist score 0–5; 5 = highly suspicious for malignancy |
| **Subtlety** | How obvious the finding is (1=barely visible, 5=very obvious) |
| **Breast Density** | Tissue density 1–4; higher density can mask lesions |
| **Abnormality Type** | Calcification or Mass |
| **Calc Type Risk** | Risk level of calcification morphology (Pleomorphic=3, Amorphous=2…) |
| **Calc Distribution** | Spatial distribution risk (Linear=3, Clustered=2…) |
| **Mass Shape Risk** | Irregular=3, Lobulated=2, Round=1 |
| **Mass Margin Risk** | Spiculated=3, Ill-defined=2, Circumscribed=0 |
| **Morphology Risk** | Sum of all above risk scores |
""")

    with col_b:
        fi_buf = feature_importance_chart(model)
        if fi_buf:
            st.markdown("### Feature Importance")
            st.image(fi_buf, use_column_width=True)

        if RESULTS_DIR.exists():
            roc_path = RESULTS_DIR / "roc_pr_curves.png"
            if roc_path.exists():
                st.markdown("### ROC / PR Curves")
                st.image(str(roc_path), use_column_width=True)

        st.markdown("""
### BI-RADS Reference
| Score | Meaning | Action |
|---|---|---|
| 0 | Incomplete | Additional imaging |
| 1 | Negative | Routine screening |
| 2 | Benign | Routine screening |
| 3 | Probably Benign | 6-month follow-up |
| 4 | Suspicious | Biopsy |
| 5 | Highly Suggestive | Biopsy |
""")
