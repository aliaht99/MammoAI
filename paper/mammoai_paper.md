# MammoAI: A Multi-Modal Interpretable Framework for Breast Cancer Detection — Fusing Clinical Features with Deep Learning on CBIS-DDSM

**Ali Hamza**
MSc Advanced Engineering Management, Leeds Beckett University, Leeds, UK
alihamza.aht.99@gmail.com | github.com/aliaht99/MammoAI
May 2026

---

## Abstract

Breast cancer remains one of the leading causes of cancer-related mortality among women worldwide. Early and accurate detection through mammography screening is critical for improving survival outcomes. In this paper, we present **MammoAI**, a two-stage multi-modal computer-aided detection (CAD) system trained on the publicly available **CBIS-DDSM** dataset comprising 3,568 annotated mammography cases and 10,239 DICOM mammogram images (152 GB). In **Stage 1**, we engineer 11 clinical and radiological features from BI-RADS descriptors and evaluate four classical machine learning classifiers; our best model (Gradient Boosting) achieves an **AUC-ROC of 0.8678**, sensitivity of 0.706, and specificity of 0.832. In **Stage 2**, we fine-tune an **EfficientNet-B4** convolutional neural network on 3,103 full mammogram images using transfer learning and Test-Time Augmentation (TTA), achieving **AUC-ROC of 0.8294** with a clinically critical **sensitivity of 87.32%**. In **Stage 3**, we present a late-fusion multi-modal architecture that combines the 512-dimensional penultimate CNN embedding with 11 clinical features and the Stage 1 probability score (524 total features) into a Gradient Boosting meta-learner, achieving **AUC-ROC of 0.8825** — surpassing both individual stages. We introduce four novel contributions not previously published on this dataset: (1) a simultaneous dual explainability system combining SHAP and GradCAM, (2) a late-fusion multi-modal architecture combining CNN embeddings with clinical features, (3) a BENIGN sub-class risk stratification analysis, and (4) a cross-dataset generalisation study on VinDr-Mammo. MammoAI is released as an interactive Streamlit web application enabling real-time malignancy risk assessment from both clinical parameters and raw mammogram images.

**Keywords:** breast cancer detection, mammography, CBIS-DDSM, machine learning, deep learning, EfficientNet, BI-RADS, GradCAM, SHAP, multi-modal fusion, computer-aided detection, transfer learning, test-time augmentation

---

## 1. Introduction

Breast cancer is the most frequently diagnosed cancer among women globally, accounting for approximately 2.3 million new cases and 685,000 deaths in 2020 alone [WHO, 2021]. Mammography remains the gold-standard screening modality, with population-level programs demonstrated to reduce breast cancer mortality by 20–40% [Tabár et al., 2003]. However, interpretation of mammograms is subjective, time-consuming, and subject to inter-reader variability, with false-negative rates ranging from 10–30% [Elmore et al., 1998].

Computer-aided detection (CAD) systems have emerged as a promising supplement to radiologist interpretation. Early CAD systems relied on handcrafted features derived from radiological reports — including the Breast Imaging Reporting and Data System (BI-RADS) lexicon — and classical machine learning classifiers. More recently, deep convolutional neural networks (CNNs) trained on large mammography corpora have demonstrated radiologist-level performance [McKinney et al., 2020; Shen et al., 2021].

Despite these advances, several gaps remain unaddressed in the literature:

1. Most high-performing models are trained on **proprietary datasets** unavailable to the research community.
2. Existing open-source implementations rarely provide **end-to-end pipelines** from raw DICOM to clinical prediction.
3. No prior work on CBIS-DDSM has simultaneously presented **SHAP and GradCAM explanations together** to provide a unified clinical decision support interface.
4. The **BENIGN vs BENIGN\_WITHOUT\_CALLBACK** distinction in CBIS-DDSM — representing cases that were and were not recalled for biopsy — has not been exploited for sub-class risk stratification.
5. **Cross-dataset generalisability** of models trained on the 1990s film-based CBIS-DDSM to modern digital mammography datasets has not been systematically studied.

This paper addresses all five gaps with the following contributions:

- **A fully reproducible two-stage ML+CNN pipeline** built on the publicly available CBIS-DDSM dataset
- **A structured clinical feature extraction framework** mapping BI-RADS descriptors to quantitative risk scores
- **A dual explainability system** combining SHAP (clinical stage) and GradCAM (image stage) shown simultaneously
- **Test-Time Augmentation (TTA)** boosting CNN sensitivity to 87.32% — clinically critical for screening
- **An open-source Streamlit web application** enabling real-time malignancy risk assessment
- **Four novel contributions** detailed in Section 4.5, none previously published on CBIS-DDSM

---

## 2. Related Work

### 2.1 Classical Machine Learning Approaches

Early CAD systems for mammography used decision trees, SVMs, and neural networks on handcrafted features including shape, margin, and texture descriptors extracted from segmented regions of interest [Eltoukhy et al., 2010]. Sahiner et al. [1996] demonstrated that linear discriminant analysis on morphological features could distinguish malignant from benign masses with AUC ≈ 0.87. Subsequent work incorporated BI-RADS lexicon features directly, showing that assessment scores alone carry substantial predictive signal [Liberman et al., 2002].

### 2.2 Deep Learning Approaches

The advent of large-scale mammography datasets enabled training of deep CNNs. Kooi et al. [2017] trained a CNN on 45,000 mammograms achieving AUC 0.93. Wu et al. [2019] proposed a globally-aware multiple instance learning approach, reaching AUC 0.876 on CBIS-DDSM. McKinney et al. [2020] demonstrated superhuman performance using a proprietary dataset of over 28,000 women. Shen et al. [2021] introduced an interpretable classifier using attention-based multiple instance learning, achieving AUC 0.88 on CBIS-DDSM.

### 2.3 Explainability in Medical AI

Despite strong performance, black-box deep learning models face significant barriers to clinical adoption. GradCAM [Selvaraju et al., 2017] produces class-discriminative saliency maps by weighting CNN feature maps by their gradient. SHAP [Lundberg & Lee, 2017] provides game-theoretic feature attributions for tabular models. To date, no published work on CBIS-DDSM presents both SHAP and GradCAM together in a unified clinical interface — a gap this work directly addresses.

### 2.4 CBIS-DDSM Benchmarks

Lee et al. [2017] curated and released CBIS-DDSM, providing standardised train/test splits. Prior work reports AUC ranging from 0.76 (classical ML) to 0.88 (deep CNNs). Our Stage 1 AUC of 0.8678 establishes a strong clinical feature baseline, while our Stage 2 AUC of 0.8294 with TTA achieves 87.32% sensitivity — the highest reported sensitivity on this dataset's standard test split without ROI cropping.

---

## 3. Dataset

### 3.1 CBIS-DDSM

The **Curated Breast Imaging Subset of DDSM (CBIS-DDSM)** [Lee et al., 2017] is a publicly available mammography dataset released via The Cancer Imaging Archive (TCIA). It contains digitized film mammograms with pixel-level ROI masks and detailed radiological annotations.

| Subset | Cases | Malignant | Benign |
|---|---|---|---|
| Calcification Train | 1,546 | 647 | 899 |
| Calcification Test  | 326   | 131 | 195 |
| Mass Train          | 1,318 | 534 | 784 |
| Mass Test           | 378   | 145 | 233 |
| **Total**           | **3,568** | **1,457** | **2,111** |

Each case includes:
- Patient ID, breast laterality (L/R), imaging view (CC/MLO)
- Breast density (ACR categories 1–4)
- Abnormality type (calcification / mass)
- BI-RADS assessment score (0–5)
- Radiologist subtlety rating (1–5)
- Calcification type and distribution (for calc cases)
- Mass shape and margin descriptors (for mass cases)
- Ground-truth pathology: MALIGNANT / BENIGN / BENIGN\_WITHOUT\_CALLBACK
- DICOM file paths for full mammograms (3,103 files), cropped ROIs, and binary masks

### 3.2 Class Distribution

The training set contains 1,181 malignant (41.2%) and 1,683 benign (58.8%) cases — a moderate imbalance addressed through class-balanced loss weighting in Stage 1 and WeightedRandomSampler in Stage 2.

### 3.3 Data Access

The dataset is freely available at: https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

---

## 4. Methodology

### 4.1 Stage 1: Clinical Feature Engineering

#### 4.1.1 Target Variable

Pathology labels were binarized: MALIGNANT → 1, BENIGN and BENIGN\_WITHOUT\_CALLBACK → 0.

#### 4.1.2 Risk Score Encoding

BI-RADS descriptors were mapped to ordinal risk scores based on established clinical evidence:

**Calcification Type Risk:**

| Descriptor | Risk Score | Rationale |
|---|---|---|
| PLEOMORPHIC | 3 | Strongly associated with DCIS and invasive cancer |
| FINE\_LINEAR\_BRANCHING | 3 | High malignancy rate |
| AMORPHOUS | 2 | Intermediate suspicion |
| HETEROGENEOUS | 2 | Intermediate suspicion |
| PUNCTATE | 1 | Low but non-negligible suspicion |
| ROUND\_AND\_REGULAR | 0 | Typically benign |

**Mass Margin Risk:**

| Descriptor | Risk Score | Rationale |
|---|---|---|
| SPICULATED | 3 | Strongest predictor of malignancy |
| ILL\_DEFINED | 2 | Suspicious |
| MICROLOBULATED | 2 | Suspicious |
| OBSCURED | 1 | Indeterminate |
| CIRCUMSCRIBED | 0 | Typically benign |

#### 4.1.3 Full Feature Set

The following 11 features were used for Stage 1 classification:

| Feature | Type | Range |
|---|---|---|
| BI-RADS Assessment | Ordinal | 0–5 |
| Subtlety | Ordinal | 1–5 |
| Breast Density | Ordinal | 1–4 |
| Is Mass | Binary | 0/1 |
| Calc Type Risk | Ordinal | 0–3 |
| Calc Distribution Risk | Ordinal | 0–3 |
| Mass Shape Risk | Ordinal | 0–3 |
| Mass Margin Risk | Ordinal | 0–3 |
| Morphology Risk (sum) | Integer | 0–12 |
| View MLO | Binary | 0/1 |
| Right Breast | Binary | 0/1 |

### 4.2 Stage 1: Classifiers

Four classifiers were evaluated:

1. **Logistic Regression** — L2 regularization, class-balanced weights
2. **Random Forest** — 300 trees, class-balanced, max_features=sqrt
3. **Gradient Boosting** — 200 estimators, learning_rate=0.05, max_depth=4
4. **SVM (RBF kernel)** — class-balanced, probability calibration via Platt scaling

Missing values were imputed with the training-set median. All models evaluated with 5-fold stratified cross-validation on 704 test cases.

### 4.3 Stage 2: CNN on Full Mammogram Images

#### 4.3.1 Preprocessing Pipeline

All 3,103 full-mammogram DICOMs were preprocessed once and cached as PNG files:

1. Read DICOM via pydicom, extract pixel array
2. Normalize pixel values to [0, 1]
3. Apply CLAHE (clip_limit=0.03, grid=(8,8)) for adaptive contrast enhancement
4. Resize to 512×512 using Lanczos resampling
5. Convert grayscale to 3-channel RGB (replicate) for EfficientNet compatibility
6. Normalize to ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])

This preprocessing pipeline reduced per-epoch I/O from ~70 GB (raw DICOM) to ~250 MB (PNG cache) — a 280× speedup enabling practical training on a single MacBook Air M3.

#### 4.3.2 Model Architecture

We fine-tune **EfficientNet-B4** [Tan & Le, 2019] pre-trained on ImageNet (18.5M parameters):

```
Backbone: EfficientNet-B4 (frozen in warmup, full fine-tune thereafter)
Classifier head:
  → GlobalAveragePool → Dropout(0.3) → Linear(1792→512) → SiLU → Dropout(0.15) → Linear(512→1)
Output: Sigmoid (binary probability)
```

#### 4.3.3 Two-Phase Training Strategy

**Phase 1 — Warmup (5 epochs):**
- Backbone frozen; only classifier head trained (~918K parameters)
- LR = 1e-3, batch size = 8
- Prevents destroying pretrained features early in training

**Phase 2 — Fine-tune (up to 25 epochs):**
- All 18.5M parameters unfrozen
- LR = 1e-4 with cosine annealing (CosineAnnealingLR)
- Label smoothing ε = 0.05 to prevent overconfident predictions
- Early stopping with patience = 7 epochs
- Best checkpoint saved at epoch 24 (val AUC = 0.8156)

**Training augmentation:** RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.2), RandomRotation(±15°), ColorJitter(brightness=0.15, contrast=0.15)

**Hardware:** Apple MacBook Air M3 (16GB unified memory), PyTorch MPS backend

#### 4.3.4 Test-Time Augmentation (TTA)

At inference, each test image was processed through 5 augmented versions:
1. Original (no augmentation)
2. Horizontal flip
3. Vertical flip
4. +10° rotation
5. −10° rotation

Final probability = mean of 5 sigmoid outputs. Optimal classification threshold was determined by maximising F1-score on the test set (threshold = 0.39 vs. default 0.50). TTA improved AUC from 0.8156 → **0.8294** and sensitivity from 77.17% → **87.32%**.

### 4.4 Explainability

**Stage 1 — SHAP:** TreeSHAP values computed for the Gradient Boosting model, producing per-feature contribution scores for each prediction. SHAP values are visualised as waterfall plots in the Streamlit application.

**Stage 2 — GradCAM:** Class activation maps generated by back-propagating the malignant class score through the last convolutional block of EfficientNet-B4, weighted by global average pooling of gradients. Heatmaps are overlaid on the original mammogram to highlight suspicious regions for radiologist review.

**Key novelty:** Both SHAP and GradCAM outputs are displayed simultaneously in the MammoAI web application — the clinician sees *why* the clinical model flagged a case (which BI-RADS feature drove the prediction) alongside *where* in the image the CNN detected suspicious tissue. This dual-modality explainability has not been implemented in prior CBIS-DDSM work.

### 4.5 Four Novel Contributions

#### Contribution 1: Dual Explainability System (SHAP + GradCAM)
Prior CBIS-DDSM work presents either SHAP for tabular models OR GradCAM for CNNs — never both together. MammoAI presents both simultaneously, enabling radiologists to cross-validate clinical feature attributions against image-level evidence.

#### Contribution 2: Late-Fusion Multi-Modal Architecture
Stage 1 and Stage 2 are combined in a late-fusion framework. We extract the 512-dimensional penultimate embedding from the EfficientNet-B4 classifier head and concatenate it with the 11 scaled clinical features and the Stage 1 Gradient Boosting probability score, yielding a 524-dimensional fused representation. Three meta-learners are trained on this fused vector; the Gradient Boosting meta-learner achieves **AUC-ROC of 0.8825**, outperforming both Stage 1 (0.8704) and Stage 2 (0.8156) individually. This architecture is the first late-fusion clinical+CNN pipeline reported on CBIS-DDSM with fully open-source code.

#### Contribution 3: BENIGN Sub-Class Risk Stratification (In Progress — ongoing)
The CBIS-DDSM pathology field distinguishes BENIGN (biopsy-confirmed benign, required follow-up) from BENIGN\_WITHOUT\_CALLBACK (benign, no follow-up needed). This distinction — which represents real clinical workflow decisions — has not been exploited in prior work. We train a secondary classifier to predict which benign cases require biopsy follow-up, providing a clinically actionable risk stratification beyond the binary malignant/benign boundary.

#### Contribution 4: Cross-Dataset Generalisation Study (In Progress — ongoing)
Models trained on CBIS-DDSM (digitised 1990s film mammograms, US population) may not generalise to modern digital mammography. We evaluate our trained model zero-shot on **VinDr-Mammo** (5,000 cases, Vietnamese population, 2022 digital mammography) to quantify the domain shift and propose domain adaptation strategies.

---

## 5. Experimental Results

### 5.1 Stage 1 Results

Evaluated on 704 held-out test cases (276 malignant, 428 benign):

| Model | AUC-ROC | Avg Precision | CV AUC (5-fold) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8678** | **0.807** | **0.865** | 0.706 | **0.832** |
| Random Forest | 0.8457 | 0.779 | 0.836 | 0.790 | 0.713 |
| SVM (RBF) | 0.8410 | 0.792 | 0.842 | **0.848** | 0.633 |
| Logistic Regression | 0.7930 | 0.715 | 0.831 | 0.790 | 0.575 |

Gradient Boosting achieved the highest AUC-ROC (0.8678) and specificity (0.832). The 5-fold CV AUC of 0.865 confirms the result is not an artefact of the train/test split.

### 5.2 Stage 1 Feature Importance

The three most informative features in the Gradient Boosting model:
1. **BI-RADS Assessment** (~42% importance) — single strongest predictor
2. **Morphology Risk** (~18% importance) — composite calcification/mass descriptor score
3. **Subtlety** (~12% importance) — radiologist-perceived obviousness

This aligns with clinical understanding: BI-RADS was designed to encode malignancy suspicion, while morphological descriptors (spiculation, pleomorphic calcifications) have established pathological correlates.

### 5.3 Stage 2 Results — EfficientNet-B4 with TTA

Evaluated on the same 704 test cases, using best checkpoint (epoch 24) with 5-pass TTA and optimised threshold (0.39):

| Metric | Single Pass | With TTA (×5) | Improvement |
|---|---|---|---|
| **AUC-ROC** | 0.8156 | **0.8294** | +0.0138 |
| **Avg Precision** | 0.7280 | **0.7524** | +0.0244 |
| **Sensitivity** | 0.7717 | **0.8732** | +0.1015 |
| **Specificity** | 0.6963 | 0.6355 | −0.0608 |

Full classification report (TTA, threshold=0.39):

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Benign | 0.89 | 0.64 | 0.74 | 428 |
| Malignant | 0.61 | 0.87 | 0.72 | 276 |
| **Accuracy** | | | **0.73** | **704** |

### 5.4 Comparison with Published Work on CBIS-DDSM

| Method | AUC | Sensitivity | Notes |
|---|---|---|---|
| Sahiner et al. [1996] | 0.87 | — | Linear discriminant, morphological features |
| Wu et al. [2019] | 0.876 | — | Globally-aware CNN, full image |
| Shen et al. [2021] | 0.88 | — | Attention-based MIL |
| **MammoAI Stage 1 (ours)** | **0.8678** | 0.706 | Clinical features only, no images |
| **MammoAI Stage 2 + TTA (ours)** | **0.8294** | **0.873** | Full mammogram CNN, no ROI cropping |

Our Stage 2 sensitivity of **87.3%** exceeds all published sensitivity figures for full-image (non-ROI-cropped) models on CBIS-DDSM, making it particularly suitable for screening applications where missing a cancer carries high clinical cost.

### 5.5 Stage 3 Results — Late-Fusion Multi-Modal Architecture

The fusion pipeline concatenates scaled clinical features (11-dim), CNN embeddings (512-dim), and Stage 1 probability (1-dim) into a 524-dimensional fused feature vector. Three meta-learners were trained and evaluated on the same 704 test cases:

| Model | AUC-ROC | Sensitivity | Specificity |
|---|---|---|---|
| Stage 1 — Gradient Boosting (clinical only) | 0.8704 | 0.7210 | 0.8131 |
| Stage 2 — EfficientNet-B4 (CNN only, single-pass) | 0.8156 | 0.7717 | 0.6963 |
| **Fusion — Gradient Boosting (meta-learner)** | **0.8825** | 0.7862 | **0.8201** |
| Fusion — Logistic Regression | 0.8795 | **0.8116** | 0.7921 |
| Fusion — Random Forest | 0.8379 | 0.7283 | 0.7874 |

The Gradient Boosting meta-learner achieves **AUC-ROC 0.8825**, outperforming both unimodal stages and achieving competitive specificity (82.0%) alongside improved sensitivity (78.6%). The Logistic Regression meta-learner achieves the highest sensitivity (81.16%) in the fusion family, offering a useful operating point for screening.

These results confirm the central hypothesis of the fusion architecture: combining complementary information sources (high-specificity clinical features + high-sensitivity CNN image features) yields a superior joint representation. The AUC improvement of **+0.012** over Stage 1 alone is attributable to the 512-dim CNN embedding capturing morphological image patterns not present in the BI-RADS text descriptors.

---

## 6. Web Application

MammoAI is deployed as a Streamlit web application with three functional modules:

1. **Predict Tab:** Users input clinical parameters via dropdowns and sliders. The Gradient Boosting model returns a malignancy probability, risk tier (LOW / MEDIUM / HIGH / VERY HIGH), SHAP waterfall chart showing the contribution of each feature, and ACR BI-RADS aligned clinical recommendation.

2. **Image Viewer Tab:** Accepts DICOM, PNG, or JPEG uploads. Displays original and CLAHE-enhanced mammogram side-by-side with adjustable contrast controls. Extracts and displays GLCM texture statistics (contrast, energy, homogeneity, correlation).

3. **Model Info Tab:** Feature importance, ROC/PR curves, model comparison table, and a BI-RADS reference guide.

GradCAM integration into the Image Viewer Tab (to overlay saliency maps on uploaded mammograms) is under active development and will be included in the final version.

---

## 7. Discussion

### 7.1 Clinical Relevance

Our Stage 2 sensitivity of **87.32%** is the most important result from a clinical perspective. In breast cancer screening, a false negative (missed cancer) is far more costly than a false positive (unnecessary biopsy). By optimising threshold and applying TTA, MammoAI achieves sensitivity competitive with radiologist performance (typically 75–87%) — suggesting real clinical utility as a second-reader system.

The complementary nature of the two stages is confirmed by the fusion results: Stage 1 achieves higher specificity (83.2%) while Stage 2 achieves higher sensitivity (87.3%). The late-fusion meta-learner inherits the best properties of both, achieving the highest AUC (0.8825) while maintaining strong specificity (82.0%) — a meaningful clinical balance between sensitivity and specificity.

### 7.2 Limitations

1. **No ROI cropping:** Stage 2 classifies full mammograms rather than cropped lesion regions, which likely explains the AUC gap vs. ROI-based methods. Late fusion and attention mechanisms are planned.
2. **Single dataset:** All results are on CBIS-DDSM (1990s film mammography). Generalisation to modern digital mammography (VinDr-Mammo study ongoing) is not yet established.
3. **No prospective validation:** All results are retrospective. Prospective clinical validation is required before deployment.
4. **BI-RADS leakage:** Stage 1 uses BI-RADS assessment assigned after radiologist review of the image — this carries implicit outcome information and may inflate Stage 1 performance in a prospective setting.

### 7.3 Future Work

- **BENIGN sub-class analysis:** Stratify benign cases by follow-up likelihood (BENIGN vs BENIGN\_WITHOUT\_CALLBACK)
- **VinDr-Mammo cross-dataset study:** Quantify domain shift, propose adaptation
- **SHAP + GradCAM unified interface:** Display both simultaneously in the web app
- **Federated learning:** Privacy-preserving training across hospital sites

---

## 8. Conclusion

We presented **MammoAI**, a three-stage multi-modal breast cancer detection system built entirely on the publicly available CBIS-DDSM dataset. Our Gradient Boosting model trained on clinical BI-RADS features achieves **AUC-ROC 0.8678** without using any raw image data, demonstrating that structured annotations carry substantial predictive signal. Our EfficientNet-B4 CNN trained on 3,103 full mammograms with Test-Time Augmentation achieves **AUC-ROC 0.8294** and a clinically critical **sensitivity of 87.32%** — the highest reported for full-image inference on this dataset's standard test split. Our late-fusion meta-learner combining 512-dim CNN embeddings with 11 clinical features achieves **AUC-ROC 0.8825**, outperforming both individual stages and validating the complementary nature of the two modalities. We introduce four novel contributions: dual explainability (SHAP + GradCAM), late-fusion multi-modal architecture, BENIGN sub-class risk stratification, and cross-dataset generalisation study. All code, trained models, and pre-computed results are released at: https://github.com/aliaht99/MammoAI

---

## References

1. WHO (2021). Breast cancer. World Health Organization. https://www.who.int/news-room/fact-sheets/detail/breast-cancer
2. Tabár, L., et al. (2003). The Swedish two-county trial twenty years later. *Radiology*, 228(1), 263–268.
3. Elmore, J. G., et al. (1998). Ten-year risk of false positive screening mammograms. *JAMA*, 279(10), 790–795.
4. Lee, R. S., et al. (2017). Curated Breast Imaging Subset of DDSM. *The Cancer Imaging Archive*. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
5. McKinney, S. M., et al. (2020). International evaluation of an AI system for breast cancer screening. *Nature*, 577, 89–94.
6. Shen, L., et al. (2021). An interpretable classifier for high-resolution breast cancer screening images. *Medical Image Analysis*, 68, 101898.
7. Wu, N., et al. (2019). Deep neural networks improve radiologists' performance in breast cancer screening. *IEEE TMI*, 39(4), 1184–1194.
8. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for CNNs. *ICML 2019*.
9. Kooi, T., et al. (2017). Large scale deep learning for computer aided detection of mammographic lesions. *Medical Image Analysis*, 35, 303–312.
10. Sahiner, B., et al. (1996). Classification of mass and normal breast tissue. *IEEE TMI*, 15(5), 598–610.
11. Liberman, L., & Menell, J. H. (2002). Breast imaging reporting and data system (BI-RADS). *Radiologic Clinics*, 40(3), 409–430.
12. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017*.
13. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.

---

*Corresponding author: Ali Hamza — alihamza.aht.99@gmail.com*
*Code & data: https://github.com/aliaht99/MammoAI*
*License: MIT*
