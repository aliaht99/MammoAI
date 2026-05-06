"""
Late-Fusion Multi-Modal Model — Novel Contribution 2 of MammoAI.

Architecture:
  Stage 1 : 11 clinical features  → Gradient Boosting probability + raw features
  Stage 2 : EfficientNet-B4 CNN   → 512-dim embedding (penultimate layer)
  Fusion  : [11 clinical + 512 CNN + 1 GB_prob] = 524 features → XGBoost meta-learner

Expected to outperform either model alone by combining:
  - High specificity of clinical model (Stage 1: 83.2%)
  - High sensitivity of CNN model    (Stage 2 TTA: 87.3%)

Usage:
    cd stage2_cnn
    python fusion.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config
from dataset import load_splits, MammogramDataset as _MammogramDataset
from model import load_checkpoint

CKPT_PATH = config.CKPT_DIR / "best_model.pth"
RESULTS_DIR = config.RESULTS_DIR / "fusion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Clinical feature engineering (mirrors src/cancer_detection.py) ──────────
CALC_TYPE_RISK = {
    "PLEOMORPHIC": 3, "FINE_LINEAR_BRANCHING": 3,
    "AMORPHOUS": 2, "HETEROGENEOUS": 2,
    "PUNCTATE": 1, "ROUND_AND_REGULAR": 0,
    "COARSE": 0, "EGGSHELL": 0, "MILK_OF_CALCIUM": 0,
    "VASCULAR": 0, "DYSTROPHIC": 0, "LUCENT_CENTER": 0,
    "SKIN": 0, "INDISTINCT": 1,
}
CALC_DIST_RISK = {
    "LINEAR": 3, "SEGMENTAL": 2, "CLUSTERED": 2,
    "REGIONAL": 1, "DIFFUSELY_SCATTERED": 0,
}
MASS_SHAPE_RISK = {
    "IRREGULAR": 3, "IRREGULAR-ARCH_DISTORTION": 3,
    "LOBULATED": 2, "LOBULATED-IRREGULAR": 2,
    "OVAL": 1, "ROUND": 1,
    "LOBULATED-OVAL": 1, "OVAL-ROUND": 1,
}
MASS_MARGIN_RISK = {
    "SPICULATED": 3, "ILL_DEFINED": 2, "MICROLOBULATED": 2,
    "OBSCURED": 1, "CIRCUMSCRIBED": 0,
    "ILL_DEFINED-SPICULATED": 3, "MICROLOBULATED-ILL_DEFINED": 2,
    "MICROLOBULATED-SPICULATED": 3, "CIRCUMSCRIBED-ILL_DEFINED": 1,
    "OBSCURED-ILL_DEFINED": 1, "OBSCURED-SPICULATED": 2,
}


def _risk(val, mapping):
    if pd.isna(val):
        return 0
    for k, v in mapping.items():
        if k in str(val).upper():
            return v
    return 0


def build_clinical_features(df: pd.DataFrame) -> np.ndarray:
    """Extract 11 clinical features — same as Stage 1 pipeline."""
    feats = pd.DataFrame()
    feats["assessment"]      = pd.to_numeric(df.get("assessment", 0), errors="coerce").fillna(0)
    feats["subtlety"]        = pd.to_numeric(df.get("subtlety", 3), errors="coerce").fillna(3)
    feats["breast_density"]  = pd.to_numeric(df.get("breast_density", 2), errors="coerce").fillna(2)
    feats["is_mass"]         = (df.get("abnormality_type", "") == "mass").astype(int)
    feats["calc_type_risk"]  = df.get("calc_type", "").apply(lambda x: _risk(x, CALC_TYPE_RISK))
    feats["calc_dist_risk"]  = df.get("calc_distribution", "").apply(lambda x: _risk(x, CALC_DIST_RISK))
    feats["mass_shape_risk"] = df.get("mass_shape", "").apply(lambda x: _risk(x, MASS_SHAPE_RISK))
    feats["mass_margin_risk"]= df.get("mass_margins", "").apply(lambda x: _risk(x, MASS_MARGIN_RISK))
    feats["morph_risk"]      = (feats["calc_type_risk"] + feats["calc_dist_risk"] +
                                feats["mass_shape_risk"] + feats["mass_margin_risk"])
    feats["view_mlo"]        = (df.get("image_view", "") == "MLO").astype(int)
    feats["is_right"]        = (df.get("left_or_right_breast", "") == "RIGHT").astype(int)
    return feats.values.astype(float)


# ── CNN embedding extractor ──────────────────────────────────────────────────
class EmbeddingExtractor(nn.Module):
    """Wraps MammoNet and returns the 512-dim penultimate embedding."""
    def __init__(self, mammonet: nn.Module):
        super().__init__()
        # backbone.features = EfficientNet convolutional body
        self.features    = mammonet.backbone.features
        self.avgpool     = mammonet.backbone.avgpool
        # classifier head up to (but not including) the final Linear(512→1)
        clf = mammonet.backbone.classifier
        # clf: [Dropout, Linear(1792→512), SiLU, Dropout, Linear(512→1)]
        self.head_embed  = nn.Sequential(*list(clf.children())[:-1])  # drop last Linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head_embed(x)
        return x  # (B, 512)


def extract_embeddings(model, df, device, desc=""):
    """Extract 512-dim CNN embeddings for all images in df."""
    extractor = EmbeddingExtractor(model).to(device)
    extractor.eval()

    ds     = _MammogramDataset(df, train=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=config.NUM_WORKERS)

    embeddings, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            emb = extractor(images.to(device)).cpu().numpy()
            embeddings.append(emb)
            labels.extend(lbls.numpy())

    embeddings = np.vstack(embeddings)
    labels     = np.array(labels)
    print(f"  {desc} embeddings: {embeddings.shape}")
    return embeddings, labels


# ── Main fusion pipeline ─────────────────────────────────────────────────────
def main():
    device = config.get_device()
    print(f"\n{'='*60}")
    print(f"  MammoAI — Late-Fusion Multi-Modal Model")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # 1. Load data
    print("[1/5] Loading datasets...")
    train_df, test_df = load_splits()

    # 2. Clinical features (Stage 1)
    print("\n[2/5] Building clinical features...")
    X_clin_train = build_clinical_features(train_df)
    X_clin_test  = build_clinical_features(test_df)
    y_train = train_df["label"].values
    y_test  = test_df["label"].values
    print(f"  Clinical features: {X_clin_train.shape[1]} features, "
          f"{len(y_train)} train / {len(y_test)} test")

    # 3. CNN embeddings (Stage 2)
    print("\n[3/5] Extracting CNN embeddings from EfficientNet-B4...")
    model, ckpt = load_checkpoint(str(CKPT_PATH), device)
    print(f"  Loaded checkpoint epoch {ckpt['epoch']} (val AUC={ckpt['val_auc']:.4f})")
    X_cnn_train, _ = extract_embeddings(model, train_df, device, "Train")
    X_cnn_test,  _ = extract_embeddings(model, test_df,  device, "Test")

    # 4. Stage 1 GB probabilities as extra feature
    print("\n[4/5] Training Stage 1 GB for probability feature...")
    scaler_clin = StandardScaler()
    X_clin_train_s = scaler_clin.fit_transform(X_clin_train)
    X_clin_test_s  = scaler_clin.transform(X_clin_test)

    gb_stage1 = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gb_stage1.fit(X_clin_train_s, y_train)
    prob_stage1_train = gb_stage1.predict_proba(X_clin_train_s)[:, 1].reshape(-1, 1)
    prob_stage1_test  = gb_stage1.predict_proba(X_clin_test_s)[:, 1].reshape(-1, 1)

    # individual stage AUCs for comparison
    auc_stage1 = roc_auc_score(y_test, gb_stage1.predict_proba(X_clin_test_s)[:, 1])
    print(f"  Stage 1 (GB clinical) AUC = {auc_stage1:.4f}")

    # Stage 2 CNN probabilities (no TTA for speed, single pass)
    model.eval()
    ds_test = _MammogramDataset(test_df, train=False)
    loader  = DataLoader(ds_test, batch_size=config.BATCH_SIZE,
                         shuffle=False, num_workers=config.NUM_WORKERS)
    prob_stage2_test = []
    with torch.no_grad():
        for images, _ in loader:
            p = torch.sigmoid(model(images.to(device))).cpu().numpy()
            prob_stage2_test.extend(p)
    prob_stage2_test = np.array(prob_stage2_test)
    auc_stage2 = roc_auc_score(y_test, prob_stage2_test)
    print(f"  Stage 2 (CNN single-pass) AUC = {auc_stage2:.4f}")

    # 5. Fuse: [clinical_features | CNN_embeddings | GB_probability]
    print("\n[5/5] Training fusion meta-learners...")
    X_fused_train = np.hstack([X_clin_train_s, X_cnn_train, prob_stage1_train])
    X_fused_test  = np.hstack([X_clin_test_s,  X_cnn_test,  prob_stage1_test])
    print(f"  Fused feature dim: {X_fused_train.shape[1]} "
          f"(11 clinical + 512 CNN + 1 GB_prob)")

    results = {}

    # Meta-learner 1: Gradient Boosting
    gb_fusion = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42)
    gb_fusion.fit(X_fused_train, y_train)
    prob_gb = gb_fusion.predict_proba(X_fused_test)[:, 1]
    results["Fusion — Gradient Boosting"] = prob_gb
    print(f"  Fusion GB    AUC = {roc_auc_score(y_test, prob_gb):.4f}")

    # Meta-learner 2: Random Forest
    rf_fusion = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_fusion.fit(X_fused_train, y_train)
    prob_rf = rf_fusion.predict_proba(X_fused_test)[:, 1]
    results["Fusion — Random Forest"] = prob_rf
    print(f"  Fusion RF    AUC = {roc_auc_score(y_test, prob_rf):.4f}")

    # Meta-learner 3: Logistic Regression (weighted average in feature space)
    scaler_fused = StandardScaler()
    X_fused_train_s = scaler_fused.fit_transform(X_fused_train)
    X_fused_test_s  = scaler_fused.transform(X_fused_test)
    lr_fusion = LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=1000, random_state=42)
    lr_fusion.fit(X_fused_train_s, y_train)
    prob_lr = lr_fusion.predict_proba(X_fused_test_s)[:, 1]
    results["Fusion — Logistic Regression"] = prob_lr
    print(f"  Fusion LR    AUC = {roc_auc_score(y_test, prob_lr):.4f}")

    # ── Final comparison table ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  FINAL COMPARISON — All Models")
    print(f"{'─'*60}")

    all_results = {
        "Stage 1 — Gradient Boosting (clinical only)":
            gb_stage1.predict_proba(X_clin_test_s)[:, 1],
        "Stage 2 — EfficientNet-B4 (CNN only)": prob_stage2_test,
        **results,
    }

    summary_rows = []
    for name, probs in all_results.items():
        auc  = roc_auc_score(y_test, probs)
        ap   = average_precision_score(y_test, probs)
        preds = (probs >= 0.5).astype(int)
        rep  = classification_report(y_test, preds, output_dict=True, zero_division=0)
        sens = rep["1"]["recall"]
        spec = rep["0"]["recall"]
        print(f"  {name:<45} AUC={auc:.4f}  Sens={sens:.4f}  Spec={spec:.4f}")
        summary_rows.append({
            "Model": name, "AUC-ROC": round(auc, 4),
            "Avg Precision": round(ap, 4),
            "Sensitivity": round(sens, 4), "Specificity": round(spec, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "fusion_comparison.csv", index=False)
    print(f"\n  Saved → {RESULTS_DIR / 'fusion_comparison.csv'}")

    # ── ROC curve comparing all models ───────────────────────────────────────
    colours = ["#0f3460", "#e94560", "#16213e", "#533483", "#2d6a4f"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for (name, probs), col in zip(all_results.items(), colours):
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        lw = 2.5 if "Fusion" in name else 1.5
        ls = "-" if "Fusion" in name else "--"
        ax.plot(fpr, tpr, color=col, lw=lw, ls=ls,
                label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — MammoAI All Stages vs Fusion Models")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fusion_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve saved → {RESULTS_DIR / 'fusion_roc.png'}")

    # ── Confusion matrix for best fusion model ────────────────────────────────
    best_name = max(results, key=lambda k: roc_auc_score(y_test, results[k]))
    best_probs = results[best_name]
    best_preds = (best_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    best_auc = roc_auc_score(y_test, best_probs)
    ax.set_title(f"Best Fusion: {best_name}\nAUC={best_auc:.4f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fusion_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {RESULTS_DIR / 'fusion_confusion.png'}")

    print(f"\n{'='*60}")
    print(f"  All fusion results saved to: {RESULTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
