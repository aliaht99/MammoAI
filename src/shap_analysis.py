"""
SHAP Explainability Analysis — Novel Contribution 1 of MammoAI.

Generates:
  results/shap/
    shap_summary_bar.png      — mean |SHAP| feature importance bar chart
    shap_summary_dot.png      — beeswarm dot plot (value + direction)
    shap_waterfall_malignant.png  — single-case waterfall, highest-risk malignant
    shap_waterfall_benign.png     — single-case waterfall, most confident benign
    shap_dependence_assessment.png — assessment vs SHAP value scatter
    shap_values.csv           — all test-set SHAP values

Usage:
    cd /Users/alihamza/Desktop/AICD
    python src/shap_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/alihamza/Desktop/AICD/manifest-ZkhPvrLo5216730872708713142")
OUT_DIR  = Path("/Users/alihamza/Desktop/AICD/results/shap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALC_TRAIN = DATA_DIR / "calc_case_description_train_set.csv"
CALC_TEST  = DATA_DIR / "calc_case_description_test_set.csv"
MASS_TRAIN = DATA_DIR / "mass_case_description_train_set.csv"
MASS_TEST  = DATA_DIR / "mass_case_description_test_set.csv"

# ── Feature engineering (mirrors cancer_detection.py) ─────────────────────────
TARGET_MAP = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}

CALC_TYPE_RISK  = {"PLEOMORPHIC": 3, "AMORPHOUS": 2, "HETEROGENEOUS": 2,
                   "FINE_LINEAR_BRANCHING": 3, "PUNCTATE": 1, "LUCENT_CENTERED": 0,
                   "ROUND_AND_REGULAR": 0, "EGGSHELL": 0, "MILK_OF_CALCIUM": 0,
                   "COARSE": 0, "LARGE_RODLIKE": 0, "DYSTROPHIC": 0}
CALC_DIST_RISK  = {"LINEAR": 3, "SEGMENTAL": 2, "REGIONAL": 1,
                   "DIFFUSELY_SCATTERED": 0, "CLUSTERED": 2}
MASS_SHAPE_RISK = {"IRREGULAR": 3, "IRREGULAR-ARCHITECTURAL_DISTORTION": 3,
                   "LOBULATED": 2, "OVAL": 1, "ROUND": 1,
                   "ARCHITECTURAL_DISTORTION": 2, "LYMPH_NODE": 0}
MASS_MARGIN_RISK= {"SPICULATED": 3, "ILL_DEFINED": 2, "OBSCURED": 1,
                   "MICROLOBULATED": 2, "CIRCUMSCRIBED": 0}

FEATURE_NAMES = [
    "BI-RADS Assessment", "Subtlety", "Breast Density", "Is Mass",
    "Calc Type Risk", "Calc Dist Risk", "Mass Shape Risk", "Mass Margin Risk",
    "Morphology Risk", "View MLO", "Right Breast",
]


def risk_score(val, mapping, default=1):
    if pd.isna(val):
        return default
    key = str(val).upper().strip()
    scores = [mapping.get(p.strip(), default) for p in key.split("-") if p.strip()]
    return max(scores) if scores else default


def load_and_engineer(split="train"):
    def _load(path):
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        for col in df.select_dtypes("object"):
            df[col] = df[col].str.strip()
        if "breast_density" not in df.columns and "breast density" in df.columns:
            df.rename(columns={"breast density": "breast_density"}, inplace=True)
        return df

    if split == "train":
        df = pd.concat([_load(CALC_TRAIN), _load(MASS_TRAIN)], ignore_index=True)
    else:
        df = pd.concat([_load(CALC_TEST), _load(MASS_TEST)], ignore_index=True)

    df["label"] = df["pathology"].map(TARGET_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    feats = pd.DataFrame()
    feats["assessment"]      = pd.to_numeric(df.get("assessment",      0), errors="coerce").fillna(0)
    feats["subtlety"]        = pd.to_numeric(df.get("subtlety",        3), errors="coerce").fillna(3)
    feats["breast_density"]  = pd.to_numeric(df.get("breast_density",  2), errors="coerce").fillna(2)
    feats["is_mass"]         = (df.get("abnormality_type", "").str.lower() == "mass").astype(int)
    feats["calc_type_risk"]  = df.get("calc_type",         "").apply(lambda x: risk_score(x, CALC_TYPE_RISK))
    feats["calc_dist_risk"]  = df.get("calc_distribution", "").apply(lambda x: risk_score(x, CALC_DIST_RISK))
    feats["mass_shape_risk"] = df.get("mass_shape",        "").apply(lambda x: risk_score(x, MASS_SHAPE_RISK))
    feats["mass_margin_risk"]= df.get("mass_margins",      "").apply(lambda x: risk_score(x, MASS_MARGIN_RISK))
    feats["morph_risk"]      = (feats["calc_type_risk"] + feats["calc_dist_risk"] +
                                feats["mass_shape_risk"] + feats["mass_margin_risk"])
    feats["view_mlo"]        = (df.get("image_view", "").str.upper() == "MLO").astype(int)
    feats["is_right"]        = (df.get("left_or_right_breast", "").str.upper() == "RIGHT").astype(int)

    return feats.values.astype(float), df["label"].values, df


def main():
    print("\n" + "="*60)
    print("  MammoAI — SHAP Explainability Analysis")
    print("="*60 + "\n")

    # 1. Load data
    print("[1/4] Loading and engineering features...")
    X_train, y_train, df_train = load_and_engineer("train")
    X_test,  y_test,  df_test  = load_and_engineer("test")
    print(f"  Train: {len(y_train)} cases | Test: {len(y_test)} cases")

    # 2. Impute + scale + train GB (same hyperparams as Stage 1)
    print("\n[2/4] Training Gradient Boosting model...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s  = scaler.transform(X_test_imp)

    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gb.fit(X_train_s, y_train)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, gb.predict_proba(X_test_s)[:, 1])
    print(f"  Test AUC = {auc:.4f}")

    # 3. SHAP TreeExplainer
    print("\n[3/4] Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X_test_s)  # shape: (n_test, 11)
    print(f"  SHAP values shape: {shap_values.shape}")

    # Save raw SHAP values to CSV
    shap_df = pd.DataFrame(shap_values, columns=FEATURE_NAMES)
    shap_df["label"] = y_test
    shap_df["pred_prob"] = gb.predict_proba(X_test_s)[:, 1]
    shap_df.to_csv(OUT_DIR / "shap_values.csv", index=False)
    print(f"  SHAP values saved → {OUT_DIR / 'shap_values.csv'}")

    # 4. Plots
    print("\n[4/4] Generating SHAP plots...")

    # ── (a) Summary bar chart — mean |SHAP| ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test_s, feature_names=FEATURE_NAMES,
                      plot_type="bar", show=False, color="#e94560")
    plt.title("SHAP Feature Importance — Mean |SHAP Value|", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar chart saved → {OUT_DIR / 'shap_summary_bar.png'}")

    # ── (b) Beeswarm dot plot — value + direction ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_s, feature_names=FEATURE_NAMES,
                      plot_type="dot", show=False)
    plt.title("SHAP Summary — Feature Value & Direction of Effect", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_dot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Beeswarm plot saved → {OUT_DIR / 'shap_summary_dot.png'}")

    # ── (c) Waterfall for highest-risk malignant case ─────────────────────────
    probs = gb.predict_proba(X_test_s)[:, 1]
    malignant_mask = y_test == 1
    idx_mal = np.where(malignant_mask)[0][np.argmax(probs[malignant_mask])]

    # SHAP 0.51+ requires scalar base_value — use float()
    base_val = float(np.array(explainer.expected_value).ravel()[0])

    explanation_mal = shap.Explanation(
        values=shap_values[idx_mal].astype(float),
        base_values=base_val,
        data=X_test_s[idx_mal].astype(float),
        feature_names=FEATURE_NAMES,
    )
    shap.waterfall_plot(explanation_mal, show=False)
    plt.title(f"SHAP Waterfall — Highest Risk Malignant Case\n"
              f"Predicted Probability: {probs[idx_mal]:.3f}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_waterfall_malignant.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Malignant waterfall saved → {OUT_DIR / 'shap_waterfall_malignant.png'}")

    # ── (d) Waterfall for most confident benign case ─────────────────────────
    benign_mask = y_test == 0
    idx_ben = np.where(benign_mask)[0][np.argmin(probs[benign_mask])]

    explanation_ben = shap.Explanation(
        values=shap_values[idx_ben].astype(float),
        base_values=base_val,
        data=X_test_s[idx_ben].astype(float),
        feature_names=FEATURE_NAMES,
    )
    shap.waterfall_plot(explanation_ben, show=False)
    plt.title(f"SHAP Waterfall — Most Confident Benign Case\n"
              f"Predicted Probability: {probs[idx_ben]:.3f}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_waterfall_benign.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Benign waterfall saved    → {OUT_DIR / 'shap_waterfall_benign.png'}")

    # ── (e) Dependence plot — BI-RADS Assessment ──────────────────────────────
    assessment_idx = 0  # first feature
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.dependence_plot(assessment_idx, shap_values, X_test_s,
                         feature_names=FEATURE_NAMES,
                         interaction_index=None,
                         ax=ax, show=False)
    ax.set_title("SHAP Dependence — BI-RADS Assessment Score", fontsize=11, fontweight="bold")
    ax.set_xlabel("BI-RADS Assessment (standardised)")
    ax.set_ylabel("SHAP Value (contribution to malignancy risk)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_dependence_assessment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dependence plot saved     → {OUT_DIR / 'shap_dependence_assessment.png'}")

    # ── Print top features ─────────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(FEATURE_NAMES, mean_abs_shap), key=lambda x: x[1], reverse=True)
    print(f"\n  Top features by mean |SHAP|:")
    for name, val in ranked:
        bar = "█" * int(val * 80 / ranked[0][1])
        print(f"    {name:<25} {val:.4f}  {bar}")

    print(f"\n  All SHAP outputs saved to: {OUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
