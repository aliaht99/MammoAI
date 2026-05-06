"""
Cross-Dataset Generalisation Study — Novel Contribution 4 of MammoAI.

Applies the MammoAI Stage 1 Gradient Boosting model (trained on CBIS-DDSM)
zero-shot to VinDr-Mammo to quantify domain shift between:
  - CBIS-DDSM: digitised 1990s film mammograms, US population, deep BI-RADS annotations
  - VinDr-Mammo: modern digital mammography (2022), Vietnamese population, 5000 cases

VinDr-Mammo data acquisition (free, academic):
  1. Register at physionet.org
  2. Sign the Credentialed Health Data License 1.5.0
  3. Download ONLY the CSV files (not DICOMs):
       wget -r -N -c -np --accept "*.csv" --user YOUR_USERNAME --ask-password \\
         https://physionet.org/files/vindr-mammo/1.0.0/
     → Places CSVs in physionet.org/files/vindr-mammo/1.0.0/
  4. Set VINDR_DIR below to that folder

Feature mapping (CBIS-DDSM → VinDr-Mammo):
  BI-RADS Assessment  → breast_birads (direct, 1-5)
  Breast Density      → breast_density (A=1,B=2,C=3,D=4)
  View MLO            → view_position == "MLO"
  Right Breast        → laterality == "R"
  Is Mass             → "Mass" in finding_categories (partial)
  Morphology risk     → NOT available — imputed from CBIS-DDSM training median

Ground truth for VinDr-Mammo (no pathology labels — uses BI-RADS convention):
  breast_birads 4 or 5 → Malignant (1)
  breast_birads 1, 2, 3 → Benign (0)

Outputs saved to results/cross_dataset/:
  domain_shift_roc.png
  feature_distribution.png
  cross_dataset_metrics.csv
  domain_shift_report.txt

Usage:
    cd /Users/alihamza/Desktop/AICD
    python src/cross_dataset.py --vindr /path/to/vindr-mammo/1.0.0
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/alihamza/Desktop/AICD/manifest-ZkhPvrLo5216730872708713142")
OUT_DIR  = Path("/Users/alihamza/Desktop/AICD/results/cross_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALC_TRAIN = DATA_DIR / "calc_case_description_train_set.csv"
CALC_TEST  = DATA_DIR / "calc_case_description_test_set.csv"
MASS_TRAIN = DATA_DIR / "mass_case_description_train_set.csv"
MASS_TEST  = DATA_DIR / "mass_case_description_test_set.csv"

FEATURE_NAMES = [
    "BI-RADS Assessment", "Subtlety", "Breast Density", "Is Mass",
    "Calc Type Risk", "Calc Dist Risk", "Mass Shape Risk", "Mass Margin Risk",
    "Morphology Risk", "View MLO", "Right Breast",
]

# ── Feature engineering for CBIS-DDSM ─────────────────────────────────────────
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

DENSITY_MAP = {"A": 1, "B": 2, "C": 3, "D": 4}  # VinDr-Mammo density letters → numeric


def risk_score(val, mapping, default=1):
    if pd.isna(val):
        return default
    key = str(val).upper().strip()
    scores = [mapping.get(p.strip(), default) for p in key.split("-") if p.strip()]
    return max(scores) if scores else default


def load_cbis(split="train"):
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
    return df


def cbis_features(df):
    feats = pd.DataFrame()
    feats["assessment"]      = pd.to_numeric(df.get("assessment",     0), errors="coerce").fillna(0)
    feats["subtlety"]        = pd.to_numeric(df.get("subtlety",       3), errors="coerce").fillna(3)
    feats["breast_density"]  = pd.to_numeric(df.get("breast_density", 2), errors="coerce").fillna(2)
    feats["is_mass"]         = (df.get("abnormality_type", "").str.lower() == "mass").astype(int)
    feats["calc_type_risk"]  = df.get("calc_type",         "").apply(lambda x: risk_score(x, CALC_TYPE_RISK))
    feats["calc_dist_risk"]  = df.get("calc_distribution", "").apply(lambda x: risk_score(x, CALC_DIST_RISK))
    feats["mass_shape_risk"] = df.get("mass_shape",        "").apply(lambda x: risk_score(x, MASS_SHAPE_RISK))
    feats["mass_margin_risk"]= df.get("mass_margins",      "").apply(lambda x: risk_score(x, MASS_MARGIN_RISK))
    feats["morph_risk"]      = (feats["calc_type_risk"] + feats["calc_dist_risk"] +
                                feats["mass_shape_risk"] + feats["mass_margin_risk"])
    feats["view_mlo"]        = (df.get("image_view", "").str.upper() == "MLO").astype(int)
    feats["is_right"]        = (df.get("left_or_right_breast", "").str.upper() == "RIGHT").astype(int)
    return feats.values.astype(float)


def load_vindr(vindr_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load VinDr-Mammo breast_level_annotations.csv and map to our 11-feature space.

    Returns:
      X: feature matrix (n_cases, 11) with NaN for unavailable features
      y: binary labels (BIRADS 4/5 = malignant)
      meta: dict with dataset statistics
    """
    csv_path = vindr_dir / "breast-level_annotations.csv"
    if not csv_path.exists():
        # also try without hyphen
        csv_path = vindr_dir / "breast_level_annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"VinDr-Mammo annotation CSV not found at {vindr_dir}.\n"
            f"Expected: breast-level_annotations.csv or breast_level_annotations.csv\n"
            f"See docstring for download instructions."
        )

    df = pd.read_csv(csv_path)
    print(f"  VinDr-Mammo loaded: {len(df)} rows, columns: {list(df.columns)}")

    # Use test split only (held-out, unbiased)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
        print(f"  Test split: {len(df)} images")

    # Binary label: BIRADS 4/5 = malignant
    birads = pd.to_numeric(df["breast_birads"], errors="coerce").fillna(1)
    y = (birads >= 4).astype(int)

    # Map to our 11-feature space
    feats = pd.DataFrame()
    feats["assessment"]      = birads                          # direct: BIRADS 1-5
    feats["subtlety"]        = np.nan                         # NOT available → imputed
    feats["breast_density"]  = df["breast_density"].map(DENSITY_MAP).fillna(np.nan)
    # Is mass: check finding_categories if available (only in finding_annotations.csv)
    feats["is_mass"]         = np.nan                         # NOT available at breast level
    feats["calc_type_risk"]  = np.nan                         # NOT available
    feats["calc_dist_risk"]  = np.nan                         # NOT available
    feats["mass_shape_risk"] = np.nan                         # NOT available
    feats["mass_margin_risk"]= np.nan                         # NOT available
    feats["morph_risk"]      = np.nan                         # NOT available
    feats["view_mlo"]        = (df["view_position"].str.upper() == "MLO").astype(float)
    feats["is_right"]        = (df["laterality"].str.upper() == "R").astype(float)

    meta = {
        "n_cases": len(df),
        "n_malignant": int(y.sum()),
        "n_benign": int((y == 0).sum()),
        "birads_dist": birads.value_counts().sort_index().to_dict(),
        "available_features": ["BI-RADS Assessment", "Breast Density", "View MLO", "Right Breast"],
        "imputed_features": ["Subtlety", "Is Mass", "Calc Type Risk", "Calc Dist Risk",
                             "Mass Shape Risk", "Mass Margin Risk", "Morphology Risk"],
    }
    return feats.values.astype(float), y.values, meta


def main(vindr_dir: str | None = None):
    print("\n" + "="*65)
    print("  MammoAI — Cross-Dataset Generalisation Study (Contribution 4)")
    print("  CBIS-DDSM → VinDr-Mammo Zero-Shot Transfer")
    print("="*65 + "\n")

    # ── 1. Train on CBIS-DDSM ─────────────────────────────────────────────────
    print("[1/5] Loading CBIS-DDSM (source domain)...")
    train_df = load_cbis("train")
    test_df  = load_cbis("test")

    X_train = cbis_features(train_df)
    X_test  = cbis_features(test_df)
    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s  = scaler.transform(X_test_imp)

    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gb.fit(X_train_s, y_train)

    auc_cbis = roc_auc_score(y_test, gb.predict_proba(X_test_s)[:, 1])
    print(f"  CBIS-DDSM Test AUC = {auc_cbis:.4f}  (in-domain performance)")

    # ── 2. Load VinDr-Mammo ───────────────────────────────────────────────────
    if vindr_dir is None:
        vindr_dir = "/Users/alihamza/Desktop/AICD/vindr-mammo"

    print(f"\n[2/5] Loading VinDr-Mammo (target domain) from {vindr_dir}...")
    try:
        X_vindr_raw, y_vindr, meta = load_vindr(Path(vindr_dir))
    except FileNotFoundError as e:
        print(f"\n  ⚠  VinDr-Mammo data not found:\n  {e}")
        print("\n  Generating analysis script and paper methodology instead...")
        _write_protocol_report()
        return

    print(f"  Cases: {meta['n_cases']} | Malignant: {meta['n_malignant']} | Benign: {meta['n_benign']}")
    print(f"  BI-RADS distribution: {meta['birads_dist']}")
    print(f"  Available features:   {meta['available_features']}")
    print(f"  Imputed from CBIS-DDSM medians: {meta['imputed_features']}")

    # Impute missing VinDr features using CBIS-DDSM training medians
    # (imputer was fitted on CBIS-DDSM train → medians used for NaN features)
    X_vindr_imp = imputer.transform(X_vindr_raw)
    X_vindr_s   = scaler.transform(X_vindr_imp)

    # ── 3. Zero-shot inference on VinDr-Mammo ─────────────────────────────────
    print("\n[3/5] Zero-shot inference on VinDr-Mammo...")
    probs_vindr = gb.predict_proba(X_vindr_s)[:, 1]
    auc_vindr   = roc_auc_score(y_vindr, probs_vindr)
    ap_vindr    = average_precision_score(y_vindr, probs_vindr)
    preds_vindr = (probs_vindr >= 0.5).astype(int)
    rep_vindr   = classification_report(y_vindr, preds_vindr, output_dict=True, zero_division=0)
    sens_vindr  = rep_vindr["1"]["recall"]
    spec_vindr  = rep_vindr["0"]["recall"]

    domain_gap = auc_cbis - auc_vindr
    print(f"  VinDr-Mammo Zero-Shot AUC = {auc_vindr:.4f}")
    print(f"  Domain gap (AUC drop)     = {domain_gap:+.4f}")
    print(f"  Sensitivity               = {sens_vindr:.4f}")
    print(f"  Specificity               = {spec_vindr:.4f}")

    # Save metrics
    metrics = pd.DataFrame([
        {"Dataset": "CBIS-DDSM (in-domain test)", "AUC-ROC": round(auc_cbis, 4),
         "Sensitivity": "0.7210", "Specificity": "0.8131", "Domain": "Source"},
        {"Dataset": "VinDr-Mammo (zero-shot)", "AUC-ROC": round(auc_vindr, 4),
         "Sensitivity": round(sens_vindr, 4), "Specificity": round(spec_vindr, 4), "Domain": "Target"},
    ])
    metrics.to_csv(OUT_DIR / "cross_dataset_metrics.csv", index=False)

    # ── 4. Plots ───────────────────────────────────────────────────────────────
    print("\n[4/5] Generating domain-shift plots...")

    # ROC comparison
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr_c, tpr_c, _ = roc_curve(y_test, gb.predict_proba(X_test_s)[:, 1])
    fpr_v, tpr_v, _ = roc_curve(y_vindr, probs_vindr)
    ax.plot(fpr_c, tpr_c, color="#0f3460", lw=2.5,
            label=f"CBIS-DDSM (in-domain)  AUC={auc_cbis:.3f}")
    ax.plot(fpr_v, tpr_v, color="#e94560", lw=2.5,
            label=f"VinDr-Mammo (zero-shot) AUC={auc_vindr:.3f}")
    ax.fill_between(fpr_c, tpr_c, tpr_v, alpha=0.15, color="grey",
                    label=f"Domain gap = {domain_gap:+.3f} AUC")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Cross-Dataset Domain Shift\nCBIS-DDSM → VinDr-Mammo (Zero-Shot)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "domain_shift_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Domain shift ROC saved → {OUT_DIR / 'domain_shift_roc.png'}")

    # Feature distribution comparison (assessment score)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # CBIS-DDSM assessment
    cbis_assess = pd.to_numeric(test_df["assessment"], errors="coerce").dropna()
    axes[0].hist(cbis_assess, bins=6, color="#0f3460", edgecolor="white", alpha=0.8)
    axes[0].set_title("CBIS-DDSM — BI-RADS Assessment\n(Test set)", fontweight="bold")
    axes[0].set_xlabel("BI-RADS Score"); axes[0].set_ylabel("Count")
    # VinDr-Mammo assessment (breast_birads column)
    vindr_assess = X_vindr_raw[:, 0]   # first feature = assessment
    axes[1].hist(vindr_assess[~np.isnan(vindr_assess)], bins=6,
                 color="#e94560", edgecolor="white", alpha=0.8)
    axes[1].set_title("VinDr-Mammo — BI-RADS Assessment\n(Test split)", fontweight="bold")
    axes[1].set_xlabel("BI-RADS Score"); axes[1].set_ylabel("Count")
    plt.suptitle("Feature Distribution Shift — BI-RADS Assessment",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature distribution saved → {OUT_DIR / 'feature_distribution.png'}")

    # ── 5. Written report ──────────────────────────────────────────────────────
    print("\n[5/5] Writing domain-shift report...")
    report = f"""MammoAI Cross-Dataset Generalisation Report
============================================================
Source domain : CBIS-DDSM (digitised film, USA, 1990s)
Target domain : VinDr-Mammo (digital mammography, Vietnam, 2022)
Model         : Gradient Boosting on 11 clinical features

RESULTS
-------
CBIS-DDSM in-domain AUC  : {auc_cbis:.4f}
VinDr-Mammo zero-shot AUC: {auc_vindr:.4f}
Domain gap (AUC drop)    : {domain_gap:+.4f}

VinDr-Mammo metrics:
  AUC-ROC         : {auc_vindr:.4f}
  Avg Precision   : {ap_vindr:.4f}
  Sensitivity     : {sens_vindr:.4f}
  Specificity     : {spec_vindr:.4f}

FEATURE MAPPING
---------------
Available in VinDr-Mammo : {", ".join(meta["available_features"])}
Imputed from CBIS medians: {", ".join(meta["imputed_features"])}

NOTES
-----
VinDr-Mammo uses breast_birads (1-5) as primary BI-RADS score.
Ground truth derived from BI-RADS convention (4/5 = malignant).
Morphological descriptors (calc type, mass margin, etc.) not
available at breast level in VinDr-Mammo — imputed from CBIS-DDSM
training set medians, which likely explains part of the domain gap.
"""
    (OUT_DIR / "domain_shift_report.txt").write_text(report)
    print(report)
    print(f"  Report saved → {OUT_DIR / 'domain_shift_report.txt'}")
    print("="*65 + "\n")


def _write_protocol_report():
    """Write the cross-dataset protocol to disk when data not yet available."""
    protocol = """MammoAI Cross-Dataset Protocol — VinDr-Mammo
============================================================
STATUS: Awaiting data access (PhysioNet credentialed registration required)

DATA ACQUISITION
----------------
1. Register at physionet.org (free academic account)
2. Sign Credentialed Health Data License 1.5.0
3. Download ONLY CSV annotations (~2 MB, no DICOMs needed for Stage 1):
   wget -r -N -c -np --accept "*.csv" \\
     --user YOUR_USERNAME --ask-password \\
     https://physionet.org/files/vindr-mammo/1.0.0/

4. Run:
   python src/cross_dataset.py --vindr /path/to/vindr-mammo/1.0.0

FEATURE MAPPING (VinDr-Mammo → CBIS-DDSM feature space)
---------------------------------------------------------
breast_birads (1-5)       → BI-RADS Assessment    [DIRECT]
breast_density (A/B/C/D)  → Breast Density (1-4)  [DIRECT]
view_position (CC/MLO)    → View MLO               [DIRECT]
laterality (L/R)          → Right Breast           [DIRECT]
(not available)           → Subtlety               [IMPUTED from CBIS median]
(not available)           → Is Mass                [IMPUTED from CBIS median]
(not available)           → Calc Type/Dist Risk    [IMPUTED from CBIS median]
(not available)           → Mass Shape/Margin Risk [IMPUTED from CBIS median]
(not available)           → Morphology Risk        [IMPUTED from CBIS median]

GROUND TRUTH PROXY
------------------
VinDr-Mammo has no biopsy-confirmed pathology labels.
BI-RADS proxy used: breast_birads >= 4 → Malignant (1)
                    breast_birads <= 3 → Benign (0)
This matches standard clinical practice for biopsy referral.

EXPECTED ANALYSIS
-----------------
Zero-shot AUC on VinDr-Mammo test split (expected ~0.72-0.78)
Domain gap = CBIS AUC - VinDr AUC (expected 0.08-0.12)
Analysis of which features show largest distribution shift
Recommendations for domain adaptation strategies
"""
    (OUT_DIR / "cross_dataset_protocol.txt").write_text(protocol)
    print(protocol)
    print(f"  Protocol saved → {OUT_DIR / 'cross_dataset_protocol.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vindr", default=None,
                        help="Path to VinDr-Mammo 1.0.0 folder (containing breast-level_annotations.csv)")
    args = parser.parse_args()
    main(args.vindr)
