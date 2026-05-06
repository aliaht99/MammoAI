"""
CBIS-DDSM Breast Cancer Detection Pipeline
Dataset: Curated Breast Imaging Subset of DDSM
Target: Predict MALIGNANT vs BENIGN from clinical mammography features

Features used (no raw DICOM images required):
  - BI-RADS assessment score
  - Breast density
  - Abnormality type (calcification / mass)
  - Calc type & distribution  (calcification cases)
  - Mass shape & margins       (mass cases)
  - Subtlety score
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 1. PATHS
# ──────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/alihamza/Desktop/AICD/manifest-ZkhPvrLo5216730872708713142")
OUT_DIR  = Path("/Users/alihamza/Desktop/AICD/results")
OUT_DIR.mkdir(exist_ok=True)

CALC_TRAIN  = DATA_DIR / "calc_case_description_train_set.csv"
CALC_TEST   = DATA_DIR / "calc_case_description_test_set.csv"
MASS_TRAIN  = DATA_DIR / "mass_case_description_train_set.csv"
MASS_TEST   = DATA_DIR / "mass_case_description_test_set.csv"


# ──────────────────────────────────────────────────────────────
# 2. LOAD & CLEAN
# ──────────────────────────────────────────────────────────────
def load_csv(path: Path) -> pd.DataFrame:
    """Read CSV, strip whitespace from column names & string values."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()
    return df


def load_dataset():
    calc_train = load_csv(CALC_TRAIN)
    calc_test  = load_csv(CALC_TEST)
    mass_train = load_csv(MASS_TRAIN)
    mass_test  = load_csv(MASS_TEST)

    # normalise breast_density column name (calc uses "breast_density" vs "breast density")
    for df in [calc_train, calc_test, mass_train, mass_test]:
        if "breast_density" not in df.columns and "breast density" in df.columns:
            df.rename(columns={"breast density": "breast_density"}, inplace=True)

    # add source tag
    calc_train["source"] = "calc"; calc_test["source"] = "calc"
    mass_train["source"] = "mass"; mass_test["source"] = "mass"

    train = pd.concat([calc_train, mass_train], ignore_index=True)
    test  = pd.concat([calc_test,  mass_test],  ignore_index=True)
    return train, test


# ──────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
TARGET_MAP = {
    "MALIGNANT":              1,
    "BENIGN":                 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
}

CALC_TYPE_RISK = {
    "PLEOMORPHIC": 3, "AMORPHOUS": 2, "HETEROGENEOUS": 2,
    "FINE_LINEAR_BRANCHING": 3, "PUNCTATE": 1, "LUCENT_CENTERED": 0,
    "ROUND_AND_REGULAR": 0, "EGGSHELL": 0, "MILK_OF_CALCIUM": 0,
    "COARSE": 0, "LARGE_RODLIKE": 0, "DYSTROPHIC": 0, "N/A": 1,
}

CALC_DIST_RISK = {
    "LINEAR": 3, "SEGMENTAL": 2, "REGIONAL": 1,
    "DIFFUSELY_SCATTERED": 0, "CLUSTERED": 2,
}

MASS_SHAPE_RISK = {
    "IRREGULAR": 3, "IRREGULAR-ARCHITECTURAL_DISTORTION": 3,
    "LOBULATED": 2, "OVAL": 1, "ROUND": 1, "ARCHITECTURAL_DISTORTION": 2,
    "LYMPH_NODE": 0,
}

MASS_MARGIN_RISK = {
    "SPICULATED": 3, "ILL_DEFINED": 2, "OBSCURED": 1,
    "MICROLOBULATED": 2, "CIRCUMSCRIBED": 0,
}


def risk_score(value: object, mapping: dict, default: int = 1) -> int:
    if pd.isna(value):
        return default
    key = str(value).upper().strip()
    # handle compound values like "PUNCTATE-PLEOMORPHIC"
    parts = key.replace("-", "_").split("_AND_")
    scores = [mapping.get(p.strip(), default) for p in parts if p.strip()]
    return max(scores) if scores else default


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # binary target
    df["label"] = df["pathology"].map(TARGET_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # BI-RADS assessment (ordinal 0-5, key predictor)
    df["assessment"] = pd.to_numeric(df["assessment"], errors="coerce")

    # subtlety (1-5, higher = more obvious)
    df["subtlety"] = pd.to_numeric(df["subtlety"], errors="coerce")

    # breast density (1-4)
    df["breast_density"] = pd.to_numeric(df["breast_density"], errors="coerce")

    # abnormality type: mass=1, calcification=0
    df["is_mass"] = (df["abnormality_type"].str.lower() == "mass").astype(int)

    # risk scores from radiological descriptors
    df["calc_type_risk"] = df.get("calc_type", pd.Series(dtype=str)).apply(
        lambda x: risk_score(x, CALC_TYPE_RISK)
    )
    df["calc_dist_risk"] = df.get("calc_distribution", pd.Series(dtype=str)).apply(
        lambda x: risk_score(x, CALC_DIST_RISK)
    )
    df["mass_shape_risk"] = df.get("mass_shape", pd.Series(dtype=str)).apply(
        lambda x: risk_score(x, MASS_SHAPE_RISK)
    )
    df["mass_margin_risk"] = df.get("mass_margins", pd.Series(dtype=str)).apply(
        lambda x: risk_score(x, MASS_MARGIN_RISK)
    )

    # combined morphology risk (whichever is relevant)
    df["morph_risk"] = df["calc_type_risk"] + df["calc_dist_risk"] + \
                       df["mass_shape_risk"] + df["mass_margin_risk"]

    # view (CC=0, MLO=1)
    df["view_mlo"] = (df["image_view"].str.upper() == "MLO").astype(int)

    # laterality (LEFT=0, RIGHT=1)
    df["is_right"] = (df["left_or_right_breast"].str.upper() == "RIGHT").astype(int)

    return df


FEATURE_COLS = [
    "assessment", "subtlety", "breast_density", "is_mass",
    "calc_type_risk", "calc_dist_risk", "mass_shape_risk", "mass_margin_risk",
    "morph_risk", "view_mlo", "is_right",
]


def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df["label"].values
    return X, y


# ──────────────────────────────────────────────────────────────
# 4. MODELS
# ──────────────────────────────────────────────────────────────
def build_models() -> dict:
    imputer = SimpleImputer(strategy="median")

    models = {
        "Logistic Regression": Pipeline([
            ("imp", imputer),
            ("scl", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imp", imputer),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imp", imputer),
            ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                               max_depth=4, random_state=42)),
        ]),
        "SVM (RBF)": Pipeline([
            ("imp", imputer),
            ("scl", StandardScaler()),
            ("clf", SVC(probability=True, class_weight="balanced", random_state=42)),
        ]),
    }
    return models


# ──────────────────────────────────────────────────────────────
# 5. EVALUATION
# ──────────────────────────────────────────────────────────────
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    auc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)
    cv   = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5),
                           scoring="roc_auc", n_jobs=-1).mean()

    report = classification_report(y_test, y_pred,
                                   target_names=["Benign", "Malignant"], output_dict=True)

    sens = report["Malignant"]["recall"]          # true positive rate
    spec = report["Benign"]["recall"]             # true negative rate

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    print(f"  CV AUC   : {cv:.4f}")
    print(f"  Sensitivity (Recall-Malignant): {sens:.4f}")
    print(f"  Specificity (Recall-Benign)   : {spec:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Benign", "Malignant"]))

    return {
        "name": name, "model": model, "auc": auc, "ap": ap, "cv_auc": cv,
        "sensitivity": sens, "specificity": spec,
        "y_pred": y_pred, "y_prob": y_prob,
    }


# ──────────────────────────────────────────────────────────────
# 6. PLOTS
# ──────────────────────────────────────────────────────────────
def plot_roc_curves(results: list, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(results)))

    # ROC
    for res, col in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, color=col,
                     label=f"{res['name']} (AUC={res['auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC Curves – All Models")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Precision-Recall
    for res, col in zip(results, colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(rec, prec, color=col,
                     label=f"{res['name']} (AP={res['ap']:.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision-Recall Curves")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] roc_pr_curves.png")


def plot_confusion_matrices(results: list, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{res['name']}\nAUC={res['auc']:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] confusion_matrices.png")


def plot_feature_importance(best_result, feature_names):
    model = best_result["model"]
    # extract RF or GB feature importance
    clf = model.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        print("[Skip] feature importance not available for this model type.")
        return

    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh([feature_names[i] for i in idx[::-1]],
                   imp[idx[::-1]], color="steelblue")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title(f"Feature Importance – {best_result['name']}", fontsize=12)
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] feature_importance.png")


def plot_data_summary(train_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("CBIS-DDSM Dataset Overview", fontsize=14, fontweight="bold")

    # pathology distribution
    ax = axes[0, 0]
    train_df["pathology"].value_counts().plot.bar(ax=ax, color=["#2196F3", "#F44336", "#4CAF50"])
    ax.set_title("Pathology Distribution")
    ax.set_xlabel(""); ax.tick_params(rotation=15)

    # BI-RADS assessment vs pathology
    ax = axes[0, 1]
    for path, grp in train_df.groupby("pathology"):
        ax.hist(grp["assessment"].dropna(), bins=range(1, 7), alpha=0.6, label=path)
    ax.set_title("BI-RADS Assessment by Pathology")
    ax.set_xlabel("Assessment Score"); ax.legend(fontsize=8)

    # breast density
    ax = axes[0, 2]
    for path, grp in train_df.groupby("pathology"):
        ax.hist(grp["breast_density"].dropna(), bins=range(1, 6), alpha=0.6, label=path)
    ax.set_title("Breast Density by Pathology")
    ax.set_xlabel("Density (1-4)"); ax.legend(fontsize=8)

    # subtlety
    ax = axes[1, 0]
    for path, grp in train_df.groupby("pathology"):
        ax.hist(grp["subtlety"].dropna(), bins=range(1, 7), alpha=0.6, label=path)
    ax.set_title("Subtlety by Pathology")
    ax.set_xlabel("Subtlety (1-5)"); ax.legend(fontsize=8)

    # abnormality type
    ax = axes[1, 1]
    ct = train_df.groupby(["abnormality_type", "pathology"]).size().unstack(fill_value=0)
    ct.plot.bar(ax=ax)
    ax.set_title("Abnormality Type vs Pathology")
    ax.set_xlabel(""); ax.tick_params(rotation=15); ax.legend(fontsize=8)

    # morph risk score
    ax = axes[1, 2]
    ax.scatter(train_df["morph_risk"], train_df["assessment"],
               c=train_df["label"], cmap="RdYlGn_r", alpha=0.4, s=15)
    ax.set_title("Morphology Risk vs Assessment\n(green=Benign, red=Malignant)")
    ax.set_xlabel("Morphology Risk Score"); ax.set_ylabel("BI-RADS Assessment")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "data_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] data_overview.png")


def plot_model_comparison(results: list):
    names  = [r["name"] for r in results]
    aucs   = [r["auc"] for r in results]
    senss  = [r["sensitivity"] for r in results]
    specs  = [r["specificity"] for r in results]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w,   aucs,  w, label="AUC-ROC",     color="#2196F3")
    ax.bar(x,       senss, w, label="Sensitivity",  color="#F44336")
    ax.bar(x + w,   specs, w, label="Specificity",  color="#4CAF50")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.9, ls="--", color="gray", lw=1, label="0.90 target")
    ax.set_title("Model Comparison", fontsize=13)
    ax.legend()
    ax.set_ylabel("Score")
    for i, (a, s, sp) in enumerate(zip(aucs, senss, specs)):
        ax.text(i - w, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i,     s + 0.01, f"{s:.2f}", ha="center", fontsize=8)
        ax.text(i + w, sp+ 0.01, f"{sp:.2f}",ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] model_comparison.png")


# ──────────────────────────────────────────────────────────────
# 7. RISK SCORING FOR NEW PATIENTS
# ──────────────────────────────────────────────────────────────
def predict_risk(model, patient: dict) -> dict:
    """
    Score a new patient from clinical features.

    patient example:
      {
        "assessment": 4,
        "subtlety": 3,
        "breast_density": 3,
        "is_mass": 1,
        "calc_type_risk": 0,
        "calc_dist_risk": 0,
        "mass_shape_risk": 3,
        "mass_margin_risk": 3,
        "morph_risk": 6,
        "view_mlo": 1,
        "is_right": 0,
      }
    """
    row = pd.DataFrame([patient])[FEATURE_COLS]
    prob  = model.predict_proba(row)[0, 1]
    label = "MALIGNANT" if prob >= 0.5 else "BENIGN"

    risk_level = ("LOW"    if prob < 0.30 else
                  "MEDIUM" if prob < 0.60 else
                  "HIGH"   if prob < 0.80 else
                  "VERY HIGH")

    return {"probability_malignant": round(prob, 4),
            "prediction": label, "risk_level": risk_level}


# ──────────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CBIS-DDSM Breast Cancer Detection Pipeline")
    print("=" * 60)

    # --- load data ---
    print("\n[1/5] Loading datasets...")
    train_raw, test_raw = load_dataset()
    print(f"  Train rows: {len(train_raw)}  | Test rows: {len(test_raw)}")

    # --- feature engineering ---
    print("\n[2/5] Engineering features...")
    train_df = engineer_features(train_raw)
    test_df  = engineer_features(test_raw)

    print(f"  Train samples: {len(train_df)} "
          f"| Malignant: {train_df['label'].sum()} "
          f"| Benign: {(train_df['label']==0).sum()}")
    print(f"  Test  samples: {len(test_df)} "
          f"| Malignant: {test_df['label'].sum()} "
          f"| Benign: {(test_df['label']==0).sum()}")

    X_train, y_train = get_X_y(train_df)
    X_test,  y_test  = get_X_y(test_df)

    # --- data visualisation ---
    print("\n[3/5] Generating data overview plots...")
    plot_data_summary(train_df)

    # --- train & evaluate models ---
    print("\n[4/5] Training and evaluating models...")
    models  = build_models()
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(res)

    # --- plots ---
    print("\n[5/5] Saving evaluation plots...")
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_model_comparison(results)

    best = max(results, key=lambda r: r["auc"])
    print(f"\n  Best model by AUC-ROC: {best['name']} ({best['auc']:.4f})")
    plot_feature_importance(best, FEATURE_COLS)

    # --- example risk prediction ---
    print("\n" + "=" * 60)
    print("  EXAMPLE: Risk Assessment for New Patient Cases")
    print("=" * 60)

    example_patients = [
        {
            "description": "High-risk: spiculated mass, BI-RADS 5",
            "features": dict(assessment=5, subtlety=4, breast_density=3, is_mass=1,
                             calc_type_risk=0, calc_dist_risk=0,
                             mass_shape_risk=3, mass_margin_risk=3,
                             morph_risk=6, view_mlo=1, is_right=0),
        },
        {
            "description": "Moderate risk: amorphous clustered calcs, BI-RADS 3",
            "features": dict(assessment=3, subtlety=2, breast_density=2, is_mass=0,
                             calc_type_risk=2, calc_dist_risk=2,
                             mass_shape_risk=0, mass_margin_risk=0,
                             morph_risk=4, view_mlo=0, is_right=1),
        },
        {
            "description": "Low-risk: round/regular calcs, BI-RADS 2, low density",
            "features": dict(assessment=2, subtlety=1, breast_density=1, is_mass=0,
                             calc_type_risk=0, calc_dist_risk=0,
                             mass_shape_risk=0, mass_margin_risk=0,
                             morph_risk=0, view_mlo=0, is_right=0),
        },
    ]

    for p in example_patients:
        risk = predict_risk(best["model"], p["features"])
        print(f"\n  Patient: {p['description']}")
        print(f"  → Malignancy probability : {risk['probability_malignant']:.1%}")
        print(f"  → Prediction             : {risk['prediction']}")
        print(f"  → Risk level             : {risk['risk_level']}")

    # --- summary table ---
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame([{
        "Model": r["name"],
        "AUC-ROC": f"{r['auc']:.4f}",
        "Avg Precision": f"{r['ap']:.4f}",
        "CV AUC (5-fold)": f"{r['cv_auc']:.4f}",
        "Sensitivity": f"{r['sensitivity']:.4f}",
        "Specificity": f"{r['specificity']:.4f}",
    } for r in results])
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "model_summary.csv", index=False)
    print(f"\n[Saved] model_summary.csv")

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
