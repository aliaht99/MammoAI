"""
Multi-Dataset Stage 1 Training — MammoAI
=========================================
Combines CBIS-DDSM (3,568 cases) + VinDr-Mammo (20,000 images) for a more
robust Gradient Boosting clinical model.

VinDr feature mapping:
  breast_birads  ("BI-RADS N") → assessment
  breast_density ("DENSITY A-D") → breast_density (A=1,B=2,C=3,D=4)
  view_position  (CC/MLO) → view_mlo
  laterality     (L/R)    → is_right
  Missing 7 features      → imputed from CBIS-DDSM training medians

Ground truth:
  CBIS-DDSM : pathology column (MALIGNANT=1)
  VinDr     : BI-RADS ≥ 4 = malignant (1), else benign (0)

Outputs:
  models/gb_multi_dataset.pkl       — trained pipeline
  results/multi_dataset/            — comparison plots + metrics

Usage:
    cd /Users/alihamza/Desktop/AICD
    python src/train_multi_dataset.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/Users/alihamza/Desktop/AICD")
DATA_DIR   = BASE_DIR / "manifest-ZkhPvrLo5216730872708713142"
VINDR_DIR  = BASE_DIR / "vindr-mammo"
MODELS_DIR = BASE_DIR / "models"
OUT_DIR    = BASE_DIR / "results" / "multi_dataset"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CBIS-DDSM CSVs
CBIS_CALC_TRAIN = DATA_DIR / "calc_case_description_train_set.csv"
CBIS_CALC_TEST  = DATA_DIR / "calc_case_description_test_set.csv"
CBIS_MASS_TRAIN = DATA_DIR / "mass_case_description_train_set.csv"
CBIS_MASS_TEST  = DATA_DIR / "mass_case_description_test_set.csv"

# VinDr-Mammo CSV
VINDR_CSV = VINDR_DIR / "breast-level_annotations.csv"

# ── Risk mappings ──────────────────────────────────────────────────────────
CALC_TYPE_RISK   = {"PLEOMORPHIC": 3, "AMORPHOUS": 2, "HETEROGENEOUS": 2,
                    "FINE_LINEAR_BRANCHING": 3, "PUNCTATE": 1, "LUCENT_CENTERED": 0,
                    "ROUND_AND_REGULAR": 0, "EGGSHELL": 0, "MILK_OF_CALCIUM": 0,
                    "COARSE": 0, "LARGE_RODLIKE": 0, "DYSTROPHIC": 0}
CALC_DIST_RISK   = {"LINEAR": 3, "SEGMENTAL": 2, "REGIONAL": 1,
                    "DIFFUSELY_SCATTERED": 0, "CLUSTERED": 2}
MASS_SHAPE_RISK  = {"IRREGULAR": 3, "IRREGULAR-ARCHITECTURAL_DISTORTION": 3,
                    "LOBULATED": 2, "OVAL": 1, "ROUND": 1,
                    "ARCHITECTURAL_DISTORTION": 2, "LYMPH_NODE": 0}
MASS_MARGIN_RISK = {"SPICULATED": 3, "ILL_DEFINED": 2, "OBSCURED": 1,
                    "MICROLOBULATED": 2, "CIRCUMSCRIBED": 0}

FEATURE_NAMES = [
    "BI-RADS Assessment", "Subtlety", "Breast Density", "Is Mass",
    "Calc Type Risk", "Calc Dist Risk", "Mass Shape Risk", "Mass Margin Risk",
    "Morphology Risk", "View MLO", "Right Breast",
]
FEATURE_COLS = [
    "assessment", "subtlety", "breast_density", "is_mass",
    "calc_type_risk", "calc_dist_risk", "mass_shape_risk", "mass_margin_risk",
    "morph_risk", "view_mlo", "is_right",
]

TARGET_MAP = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}


def risk_score(val, mapping, default=1):
    if pd.isna(val):
        return default
    key = str(val).upper().strip()
    scores = [mapping.get(p.strip(), default) for p in key.split("-") if p.strip()]
    return max(scores) if scores else default


# ── CBIS-DDSM loader ───────────────────────────────────────────────────────
def load_cbis(split="train"):
    def _read(path):
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        for col in df.select_dtypes("object"):
            df[col] = df[col].str.strip()
        if "breast_density" not in df.columns and "breast density" in df.columns:
            df.rename(columns={"breast density": "breast_density"}, inplace=True)
        return df

    if split == "train":
        df = pd.concat([_read(CBIS_CALC_TRAIN), _read(CBIS_MASS_TRAIN)], ignore_index=True)
    else:
        df = pd.concat([_read(CBIS_CALC_TEST), _read(CBIS_MASS_TEST)], ignore_index=True)

    df["label"] = df["pathology"].map(TARGET_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    feats = pd.DataFrame()
    feats["assessment"]       = pd.to_numeric(df.get("assessment", 0), errors="coerce").fillna(0)
    feats["subtlety"]         = pd.to_numeric(df.get("subtlety", 3), errors="coerce").fillna(3)
    feats["breast_density"]   = pd.to_numeric(df.get("breast_density", 2), errors="coerce").fillna(2)
    feats["is_mass"]          = (df.get("abnormality_type", "").str.lower() == "mass").astype(int)
    feats["calc_type_risk"]   = df.get("calc_type", "").apply(lambda x: risk_score(x, CALC_TYPE_RISK))
    feats["calc_dist_risk"]   = df.get("calc_distribution", "").apply(lambda x: risk_score(x, CALC_DIST_RISK))
    feats["mass_shape_risk"]  = df.get("mass_shape", "").apply(lambda x: risk_score(x, MASS_SHAPE_RISK))
    feats["mass_margin_risk"] = df.get("mass_margins", "").apply(lambda x: risk_score(x, MASS_MARGIN_RISK))
    feats["morph_risk"]       = (feats["calc_type_risk"] + feats["calc_dist_risk"] +
                                  feats["mass_shape_risk"] + feats["mass_margin_risk"])
    feats["view_mlo"]         = (df.get("image_view", "").str.upper() == "MLO").astype(int)
    feats["is_right"]         = (df.get("left_or_right_breast", "").str.upper() == "RIGHT").astype(int)
    feats["source"]           = "CBIS-DDSM"

    return feats.values[:, :11].astype(float), df["label"].values


# ── VinDr-Mammo loader ─────────────────────────────────────────────────────
def load_vindr(split="training", cbis_medians=None):
    if not VINDR_CSV.exists():
        print(f"  [WARN] VinDr CSV not found: {VINDR_CSV}")
        return None, None

    df = pd.read_csv(VINDR_CSV)
    df = df[df["split"] == split].copy()

    # Parse BI-RADS: "BI-RADS 2" → 2
    birads_raw = df["breast_birads"].astype(str).str.extract(r'(\d+)')[0]
    birads = pd.to_numeric(birads_raw, errors="coerce").fillna(1)

    # Parse density: "DENSITY C" → C→3
    density_letter = df["breast_density"].astype(str).str.extract(r'DENSITY\s+([A-D])')[0]
    density_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    density = density_letter.map(density_map).fillna(2)

    view_mlo = (df["view_position"].str.upper() == "MLO").astype(int)
    is_right = (df["laterality"].str.upper() == "R").astype(int)

    # Ground truth: BI-RADS ≥ 4 = malignant
    labels = (birads >= 4).astype(int).values

    # Build feature matrix — 7 missing features imputed from CBIS medians
    n = len(df)
    feats = np.zeros((n, 11), dtype=float)
    feats[:, 0] = birads.values          # assessment
    feats[:, 1] = cbis_medians[1] if cbis_medians is not None else 3   # subtlety (CBIS median)
    feats[:, 2] = density.values         # breast_density
    feats[:, 3] = cbis_medians[3] if cbis_medians is not None else 0   # is_mass
    feats[:, 4] = cbis_medians[4] if cbis_medians is not None else 1   # calc_type_risk
    feats[:, 5] = cbis_medians[5] if cbis_medians is not None else 1   # calc_dist_risk
    feats[:, 6] = cbis_medians[6] if cbis_medians is not None else 1   # mass_shape_risk
    feats[:, 7] = cbis_medians[7] if cbis_medians is not None else 1   # mass_margin_risk
    feats[:, 8] = cbis_medians[8] if cbis_medians is not None else 2   # morph_risk
    feats[:, 9] = view_mlo.values        # view_mlo
    feats[:, 10] = is_right.values       # is_right

    return feats, labels


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*68)
    print("  MammoAI — Multi-Dataset Stage 1 Training")
    print("  Sources: CBIS-DDSM + VinDr-Mammo")
    print("="*68 + "\n")

    # 1. Load CBIS-DDSM
    print("[1/6] Loading CBIS-DDSM...")
    X_cbis_tr, y_cbis_tr = load_cbis("train")
    X_cbis_te, y_cbis_te = load_cbis("test")
    print(f"  CBIS train: {len(y_cbis_tr)} | test: {len(y_cbis_te)}")
    print(f"  Malignant rate — train: {y_cbis_tr.mean():.2%} | test: {y_cbis_te.mean():.2%}")

    # Compute CBIS-DDSM medians for VinDr imputation
    cbis_medians = np.median(X_cbis_tr, axis=0)

    # 2. Load VinDr-Mammo
    print("\n[2/6] Loading VinDr-Mammo...")
    X_vindr_tr, y_vindr_tr = load_vindr("training", cbis_medians)
    X_vindr_te, y_vindr_te = load_vindr("test", cbis_medians)

    if X_vindr_tr is not None:
        print(f"  VinDr train: {len(y_vindr_tr)} | test: {len(y_vindr_te)}")
        print(f"  Malignant rate — train: {y_vindr_tr.mean():.2%} | test: {y_vindr_te.mean():.2%}")
        # Combine
        X_train = np.vstack([X_cbis_tr, X_vindr_tr])
        y_train = np.concatenate([y_cbis_tr, y_vindr_tr])
        X_test  = np.vstack([X_cbis_te, X_vindr_te])
        y_test  = np.concatenate([y_cbis_te, y_vindr_te])
        source_labels = (["CBIS"] * len(y_cbis_te) + ["VinDr"] * len(y_vindr_te))
    else:
        print("  VinDr not available — using CBIS-DDSM only")
        X_train, y_train = X_cbis_tr, y_cbis_tr
        X_test,  y_test  = X_cbis_te, y_cbis_te
        source_labels = ["CBIS"] * len(y_cbis_te)

    print(f"\n  Combined train: {len(y_train)} cases (malignant: {y_train.mean():.2%})")
    print(f"  Combined test:  {len(y_test)} cases (malignant:  {y_test.mean():.2%})")

    # 3. Preprocess
    print("\n[3/6] Preprocessing (impute + scale)...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s  = scaler.transform(X_test_imp)

    # 4. Train classifiers
    print("\n[4/6] Training classifiers...")
    classifiers = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
    }

    results = {}
    rows = []
    for name, clf in classifiers.items():
        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc   = roc_auc_score(y_test, probs)
        ap    = average_precision_score(y_test, probs)
        rep   = classification_report(y_test, preds, output_dict=True, zero_division=0)
        sens  = rep.get("1", {}).get("recall", 0.0)
        spec  = rep.get("0", {}).get("recall", 0.0)
        cv    = cross_val_score(clf, X_train_s, y_train, cv=StratifiedKFold(5),
                                scoring="roc_auc", n_jobs=-1).mean()
        results[name] = {"clf": clf, "probs": probs, "auc": auc, "cv": cv}
        rows.append({"Model": name, "AUC-ROC": round(auc, 4), "CV-AUC": round(cv, 4),
                     "Sensitivity": round(sens, 4), "Specificity": round(spec, 4),
                     "Avg Precision": round(ap, 4)})
        print(f"  {name:<25} AUC={auc:.4f}  CV={cv:.4f}  Sens={sens:.3f}  Spec={spec:.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "multi_dataset_metrics.csv", index=False)

    # 5. Save best model (GB) as production pipeline
    print("\n[5/6] Saving models...")
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_clf  = results[best_name]["clf"]
    print(f"  Best model: {best_name} (AUC={results[best_name]['auc']:.4f})")

    pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler",  scaler),
        ("clf",     best_clf),
    ])

    multi_path = MODELS_DIR / "gb_multi_dataset.pkl"
    with open(multi_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  Saved → {multi_path}")

    # Also save metadata for the app
    meta = {
        "feature_names": FEATURE_NAMES,
        "feature_cols": FEATURE_COLS,
        "cbis_medians": cbis_medians.tolist(),
        "datasets": ["CBIS-DDSM", "VinDr-Mammo"] if X_vindr_tr is not None else ["CBIS-DDSM"],
        "train_size": int(len(y_train)),
        "auc": results[best_name]["auc"],
        "malignant_rate": float(y_train.mean()),
    }
    pd.Series(meta).to_json(MODELS_DIR / "gb_multi_dataset_meta.json")
    print(f"  Saved → {MODELS_DIR / 'gb_multi_dataset_meta.json'}")

    # 6. Plots
    print("\n[6/6] Generating plots...")
    colours = ["#e94560", "#0f3460", "#2d6a4f"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC comparison
    ax = axes[0]
    for (name, res), col in zip(results.items(), colours):
        fpr, tpr, _ = roc_curve(y_test, res["probs"])
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"ROC — Multi-Dataset Model\n(CBIS-DDSM {len(y_cbis_te):,} + VinDr {len(y_vindr_te) if X_vindr_tr is not None else 0:,} test cases)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Per-source ROC (GB only)
    ax2 = axes[1]
    gb_probs = results["Gradient Boosting"]["probs"]
    for src, col, lbl in [("CBIS", "#e94560", "CBIS-DDSM"), ("VinDr", "#0f3460", "VinDr-Mammo")]:
        mask = np.array(source_labels) == src
        if mask.sum() > 10 and y_test[mask].sum() > 0:
            fpr, tpr, _ = roc_curve(y_test[mask], gb_probs[mask])
            auc_src = roc_auc_score(y_test[mask], gb_probs[mask])
            ax2.plot(fpr, tpr, color=col, lw=2, label=f"{lbl} (AUC={auc_src:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="Per-Source ROC — Gradient Boosting")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "multi_dataset_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC saved → {OUT_DIR / 'multi_dataset_roc.png'}")

    # Metrics table plot
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0f3460")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4ff")
    ax.set_title("Multi-Dataset Model Metrics", fontweight="bold", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "metrics_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Metrics table → {OUT_DIR / 'metrics_table.png'}")

    print(f"\n  All outputs → {OUT_DIR}")
    print("="*68 + "\n")


if __name__ == "__main__":
    main()
