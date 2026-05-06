"""
BENIGN Sub-Class Risk Stratification — Novel Contribution 3 of MammoAI.

Problem: CBIS-DDSM pathology has THREE classes:
  - MALIGNANT
  - BENIGN                   (biopsy-confirmed benign, required follow-up)
  - BENIGN_WITHOUT_CALLBACK  (benign, no follow-up needed)

This distinction represents real clinical workflow decisions — cases marked
BENIGN_WITHOUT_CALLBACK were deemed low enough suspicion to not recall
the patient for biopsy. We train a secondary classifier to predict which
benign-outcome cases actually required biopsy (BENIGN vs BENIGN_WITHOUT_CALLBACK).

Outputs saved to results/benign_subclass/:
  roc_curve.png
  confusion_matrix.png
  shap_summary.png
  subclass_metrics.csv

Usage:
    cd /Users/alihamza/Desktop/AICD
    python src/benign_subclass.py
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/alihamza/Desktop/AICD/manifest-ZkhPvrLo5216730872708713142")
OUT_DIR  = Path("/Users/alihamza/Desktop/AICD/results/benign_subclass")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALC_TRAIN = DATA_DIR / "calc_case_description_train_set.csv"
CALC_TEST  = DATA_DIR / "calc_case_description_test_set.csv"
MASS_TRAIN = DATA_DIR / "mass_case_description_train_set.csv"
MASS_TEST  = DATA_DIR / "mass_case_description_test_set.csv"

# ── Feature engineering ────────────────────────────────────────────────────────
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


def load_raw(split="train"):
    """Load data keeping ALL three pathology classes."""
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
    return df


def engineer_features(df: pd.DataFrame) -> np.ndarray:
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


def main():
    print("\n" + "="*65)
    print("  MammoAI — BENIGN Sub-Class Risk Stratification (Contribution 3)")
    print("="*65 + "\n")

    # 1. Load raw data (3 classes)
    train_raw = load_raw("train")
    test_raw  = load_raw("test")

    # Class counts
    print("[1/5] Full dataset class distribution:")
    for df, name in [(train_raw, "Train"), (test_raw, "Test")]:
        counts = df["pathology"].value_counts()
        print(f"  {name}:")
        for cls, n in counts.items():
            print(f"    {cls:<30} {n:4d}")
    print()

    # 2. Filter to BENIGN-outcome cases only, create binary sub-label
    #    BENIGN = 1 (required biopsy follow-up — higher risk)
    #    BENIGN_WITHOUT_CALLBACK = 0 (no follow-up needed — lower risk)
    def filter_benign(df):
        mask = df["pathology"].isin(["BENIGN", "BENIGN_WITHOUT_CALLBACK"])
        sub  = df[mask].copy()
        sub["sub_label"] = (sub["pathology"] == "BENIGN").astype(int)
        return sub

    train_sub = filter_benign(train_raw)
    test_sub  = filter_benign(test_raw)

    print("[2/5] BENIGN sub-class dataset (BENIGN=1 vs BENIGN_WITHOUT_CALLBACK=0):")
    print(f"  Train: {len(train_sub)} cases  — BENIGN={train_sub['sub_label'].sum()}, "
          f"BENIGN_WC={(train_sub['sub_label']==0).sum()}")
    print(f"  Test:  {len(test_sub)} cases   — BENIGN={test_sub['sub_label'].sum()}, "
          f"BENIGN_WC={(test_sub['sub_label']==0).sum()}\n")

    # 3. Feature engineering + preprocessing
    print("[3/5] Building features and preprocessing...")
    X_train = engineer_features(train_sub)
    X_test  = engineer_features(test_sub)
    y_train = train_sub["sub_label"].values
    y_test  = test_sub["sub_label"].values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s  = scaler.transform(X_test_imp)

    # 4. Train classifiers
    print("[4/5] Training sub-class classifiers...")
    models = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
    }

    results = {}
    summary_rows = []
    for name, clf in models.items():
        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc   = roc_auc_score(y_test, probs)
        ap    = average_precision_score(y_test, probs)
        rep   = classification_report(y_test, preds, output_dict=True, zero_division=0)
        sens  = rep["1"]["recall"]   # sensitivity for BENIGN
        spec  = rep["0"]["recall"]   # specificity (BENIGN_WC correctly identified)
        results[name] = (clf, probs, preds)
        summary_rows.append({"Model": name, "AUC-ROC": round(auc, 4),
                              "Avg Precision": round(ap, 4),
                              "Sensitivity": round(sens, 4), "Specificity": round(spec, 4)})
        print(f"  {name:<25} AUC={auc:.4f}  Sens={sens:.4f}  Spec={spec:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "subclass_metrics.csv", index=False)

    # 5. Plots for best model (GB)
    print("\n[5/5] Generating plots...")
    best_name = max(results, key=lambda k: roc_auc_score(y_test, results[k][1]))
    best_clf, best_probs, best_preds = results[best_name]
    best_auc = roc_auc_score(y_test, best_probs)

    # ROC curve — all 3 models
    colours = ["#e94560", "#0f3460", "#2d6a4f"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, (_, probs, _)), col in zip(results.items(), colours):
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC — BENIGN Sub-Class Stratification\n"
                 "(BENIGN=requires biopsy vs BENIGN_WITHOUT_CALLBACK)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve saved → {OUT_DIR / 'roc_curve.png'}")

    # Confusion matrix
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Biopsy\n(B_WC)", "Biopsy\n(BENIGN)"]).plot(
        ax=ax, colorbar=False, cmap="Purples")
    ax.set_title(f"Best Model: {best_name}\nAUC={best_auc:.4f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {OUT_DIR / 'confusion_matrix.png'}")

    # SHAP summary for GB sub-class model
    gb_clf = results["Gradient Boosting"][0]
    explainer   = shap.TreeExplainer(gb_clf)
    shap_values = explainer.shap_values(X_test_s)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test_s, feature_names=FEATURE_NAMES,
                      plot_type="bar", show=False, color="#533483")
    plt.title("SHAP — BENIGN Sub-Class Model Feature Importance\n"
              "(which features predict biopsy-required vs. no follow-up?)",
              fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary saved → {OUT_DIR / 'shap_summary.png'}")

    # Print classification report
    print(f"\n  Best model ({best_name}) Classification Report:")
    print(classification_report(y_test, best_preds,
                                target_names=["BENIGN_WC (0)", "BENIGN (1)"],
                                zero_division=0))

    # Print SHAP top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(FEATURE_NAMES, mean_abs_shap), key=lambda x: x[1], reverse=True)
    print(f"  Top features for biopsy-need prediction (|SHAP|):")
    for name, val in ranked[:5]:
        bar = "█" * int(val * 50 / ranked[0][1])
        print(f"    {name:<25} {val:.4f}  {bar}")

    print(f"\n  All outputs saved to: {OUT_DIR}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
