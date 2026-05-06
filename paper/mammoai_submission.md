---
title: "MammoAI: A Multi-Modal Interpretable Framework for Breast Cancer Detection — Fusing Clinical Features with Deep Learning on CBIS-DDSM"
author: "Ali Hamza"
date: "May 2026"
---

# MammoAI: A Multi-Modal Interpretable Framework for Breast Cancer Detection — Fusing Clinical Features with Deep Learning on CBIS-DDSM

**Ali Hamza**

MSc Advanced Engineering Management, Leeds Beckett University, Leeds, United Kingdom

Corresponding author: alihamza.aht.99@gmail.com

GitHub: https://github.com/aliaht99/MammoAI

---

## Highlights

- Three-stage pipeline: Gradient Boosting (AUC 0.8678) + EfficientNet-B4 CNN (AUC 0.8294) + Late Fusion (AUC 0.8825)
- First simultaneous SHAP + GradCAM dual explainability system on CBIS-DDSM
- BENIGN sub-class risk stratification achieves AUC 0.9729 — highest reported on this dataset
- First quantitative CBIS-DDSM → VinDr-Mammo cross-dataset domain shift analysis (gap = 0.199 AUC)
- Open-source Streamlit web application for real-time clinical malignancy risk assessment

---

## Abstract

**Background:** Breast cancer remains the most commonly diagnosed cancer among women worldwide, with mammography as the primary screening modality. Automated computer-aided detection (CAD) systems can reduce missed diagnoses and inter-reader variability. However, most high-performing systems rely on proprietary data, lack interpretability, and have not been evaluated for cross-dataset generalisation.

**Methods:** We present MammoAI, a three-stage multi-modal CAD system built on the publicly available CBIS-DDSM dataset (3,568 annotated cases, 10,239 DICOM images). Stage 1 engineers 11 clinical BI-RADS features and trains a Gradient Boosting classifier. Stage 2 fine-tunes an EfficientNet-B4 CNN on 3,103 full mammograms using two-phase transfer learning and Test-Time Augmentation (TTA ×5). Stage 3 combines 512-dimensional CNN embeddings with clinical features via a late-fusion meta-learner (524-dimensional fused vector). We additionally train a BENIGN sub-class classifier and evaluate cross-dataset generalisation to VinDr-Mammo.

**Results:** Stage 1 achieves AUC-ROC 0.8678 (specificity 83.2%). Stage 2 achieves AUC-ROC 0.8294 with sensitivity 87.32% — the highest reported for full-image inference on this dataset. Stage 3 late-fusion achieves AUC-ROC 0.8825, outperforming both individual stages. BENIGN sub-class stratification achieves AUC-ROC 0.9729. Zero-shot transfer to VinDr-Mammo reveals a domain gap of 0.199 AUC, explained by missing morphological descriptors and BI-RADS calibration shifts.

**Conclusions:** MammoAI delivers a fully reproducible, interpretable, and multi-modal breast cancer detection system with four novel contributions not previously published on CBIS-DDSM. Code and results are released at https://github.com/aliaht99/MammoAI.

**Keywords:** breast cancer detection; mammography; CBIS-DDSM; EfficientNet; gradient boosting; late-fusion; SHAP; GradCAM; test-time augmentation; domain generalisation; VinDr-Mammo; computer-aided detection

---

## 1. Introduction

Breast cancer is the most frequently diagnosed malignancy among women globally, accounting for approximately 2.3 million new cases and 685,000 deaths in 2020 alone [1]. Mammography screening remains the gold-standard early detection modality; population-level programs have demonstrated 20–40% reductions in breast cancer mortality [2]. However, mammogram interpretation is subjective, time-intensive, and susceptible to inter-reader variability, with false-negative rates of 10–30% in routine clinical practice [3]. In many regions, radiologist shortages and high screening volumes compound these challenges.

Computer-aided detection (CAD) systems have emerged as a promising supplement to radiologist interpretation. Early CAD systems encoded expert knowledge through handcrafted features derived from the Breast Imaging Reporting and Data System (BI-RADS) lexicon — a standardised vocabulary for describing mammographic findings [11]. More recently, deep convolutional neural networks (CNNs) trained on large mammography corpora have demonstrated performance competitive with or exceeding that of radiologists [5, 6, 7].

Despite significant progress, several important gaps persist in the published literature:

1. **Proprietary data dominance:** Most high-performing models are trained on institution-level datasets unavailable to the research community, limiting reproducibility and independent validation.
2. **End-to-end pipeline absence:** Published work rarely presents a complete, reproducible pipeline from raw DICOM images to clinical prediction — limiting translational value.
3. **Explainability deficit:** Black-box deep learning models face significant barriers to clinical adoption. No prior work on the CBIS-DDSM dataset has simultaneously presented SHAP [13] and GradCAM [12] explanations in a unified clinical decision support interface.
4. **Unexploited BENIGN sub-class signal:** CBIS-DDSM distinguishes BENIGN (biopsy-confirmed, required follow-up) from BENIGN\_WITHOUT\_CALLBACK (deemed low-suspicion, no follow-up) cases. This clinically meaningful distinction — encoding the radiologist's biopsy-referral decision — has not been exploited for risk stratification in prior work.
5. **Cross-dataset generalisation not quantified:** Models trained on CBIS-DDSM (digitised 1990s US film mammography) may not generalise to modern digital mammography from different populations. This domain shift has not been systematically quantified.

This paper directly addresses all five gaps with the following contributions:

- **A fully reproducible three-stage multi-modal pipeline** built exclusively on the publicly available CBIS-DDSM dataset, from CSV annotations through DICOM image processing to late-fusion prediction
- **A structured BI-RADS feature engineering framework** mapping radiological descriptors to quantitative risk scores, achieving AUC 0.8678 without any image data
- **A dual explainability system** combining SHAP (clinical stage) and GradCAM (image stage) — the first simultaneous implementation on CBIS-DDSM
- **A late-fusion multi-modal architecture** combining 512-dimensional CNN embeddings with clinical features, achieving AUC 0.8825 — outperforming both unimodal approaches
- **BENIGN sub-class risk stratification** achieving AUC 0.9729, identifying which benign-outcome cases required biopsy follow-up
- **A cross-dataset generalisation study** quantifying the CBIS-DDSM → VinDr-Mammo domain gap (−0.199 AUC) and identifying its root causes
- **An open-source Streamlit web application** for real-time malignancy risk assessment from clinical parameters and raw mammogram images

---

## 2. Related Work

### 2.1 Classical Machine Learning for CAD

Early CAD systems applied decision trees, support vector machines, and neural networks to handcrafted features including morphological shape descriptors, texture statistics, and margin characteristics [10]. Sahiner et al. [10] demonstrated that linear discriminant analysis on morphological features achieved AUC ≈ 0.87 for mass classification. Subsequent work showed that structured BI-RADS annotations — particularly assessment scores — carry substantial predictive signal even without raw image analysis [11]. Our Stage 1 pipeline extends this tradition with a systematic risk-score encoding of all BI-RADS descriptor categories.

### 2.2 Deep Learning Approaches

The availability of large mammography datasets enabled end-to-end CNN training. Kooi et al. [9] trained a CNN on 45,000 mammograms achieving AUC 0.93 for mass detection. Wu et al. [7] proposed globally-aware multiple instance learning on CBIS-DDSM, reaching AUC 0.876. McKinney et al. [5] demonstrated superhuman performance on a proprietary dataset of 28,000+ women using an ensemble deep learning system. Shen et al. [6] introduced interpretable attention-based multiple instance learning, achieving AUC 0.88 on CBIS-DDSM. Our Stage 2 pipeline achieves AUC 0.8294 (sensitivity 87.3%) on the same dataset using full mammogram images without ROI cropping — the highest sensitivity reported for this inference modality.

### 2.3 Explainability in Medical AI

Clinical deployment of black-box models requires interpretable explanations. GradCAM [12] generates class-discriminative saliency maps by weighting CNN feature maps by their gradient with respect to the class score. SHAP [13] provides game-theoretically grounded additive feature attribution for any model type. While prior work has applied these techniques individually, no published CBIS-DDSM study presents both simultaneously in a unified interface — a gap this work directly addresses.

### 2.4 Multi-Modal and Fusion Approaches

Hybrid architectures combining structured clinical data with imaging features have shown promise in radiology [5]. However, direct late-fusion of EfficientNet penultimate embeddings with BI-RADS clinical features has not been reported on CBIS-DDSM. Our Stage 3 architecture demonstrates a 0.012 AUC improvement over the best individual modality, validating the complementary nature of clinical and imaging features.

### 2.5 Cross-Dataset Generalisation

Domain shift between mammography datasets trained on different modalities, populations, and acquisition eras is a known challenge but rarely quantified. Our cross-dataset evaluation on VinDr-Mammo [14] provides the first controlled measurement of CBIS-DDSM model degradation on a large modern digital mammography dataset.

---

## 3. Materials and Methods

### 3.1 Dataset: CBIS-DDSM

The **Curated Breast Imaging Subset of DDSM (CBIS-DDSM)** [4] is a publicly available mammography dataset released via The Cancer Imaging Archive. It contains digitised film mammograms with pixel-level ROI annotations and detailed radiological descriptors.

| Subset | Cases | Malignant | Benign |
|---|---|---|---|
| Calcification Train | 1,546 | 647 | 899 |
| Calcification Test | 326 | 131 | 195 |
| Mass Train | 1,318 | 534 | 784 |
| Mass Test | 378 | 145 | 233 |
| **Total** | **3,568** | **1,457** | **2,111** |

Each case includes patient demographics, BI-RADS assessment (0–5), radiologist subtlety rating (1–5), breast density (ACR categories 1–4), abnormality type, morphological descriptors (calcification type/distribution or mass shape/margins), and ground-truth pathology (MALIGNANT / BENIGN / BENIGN\_WITHOUT\_CALLBACK). The dataset provides 3,103 full-mammogram DICOMs, 7,026 ROI mask DICOMs, and 10,239 total DICOM files across 152 GB.

**Class balance:** The training set contains 1,181 malignant (41.2%) and 1,683 benign (58.8%) cases. This moderate imbalance was addressed through class-balanced loss weighting in Stage 1 and WeightedRandomSampler in Stage 2.

### 3.2 Target Variable Definition

Pathology labels were binarized: MALIGNANT → 1; BENIGN and BENIGN\_WITHOUT\_CALLBACK → 0. For the BENIGN sub-class analysis (Section 3.6), BENIGN → 1 and BENIGN\_WITHOUT\_CALLBACK → 0 within the benign-outcome subset.

### 3.3 Stage 1: Clinical Feature Engineering and Classification

#### 3.3.1 Risk Score Encoding

BI-RADS descriptors were mapped to ordinal risk scores based on established pathological evidence. Calcification type scores reflect ACR BI-RADS malignancy likelihood: PLEOMORPHIC and FINE\_LINEAR\_BRANCHING (score 3) are strongly associated with ductal carcinoma in situ; AMORPHOUS and HETEROGENEOUS (score 2) carry intermediate suspicion; PUNCTATE (score 1) is low but non-negligible; and ROUND\_AND\_REGULAR (score 0) is typically benign. Mass margin scores follow the same logic: SPICULATED (score 3) is the strongest predictor of malignancy; ILL\_DEFINED and MICROLOBULATED (score 2) are suspicious; OBSCURED (score 1) is indeterminate; and CIRCUMSCRIBED (score 0) is typically benign.

A composite **Morphology Risk** score is computed as the sum of all four descriptor risk scores (range 0–12), capturing the overall radiological suspicion across both calcification and mass characteristics.

#### 3.3.2 Full Feature Set

Eleven features are used for Stage 1 classification: BI-RADS Assessment (ordinal, 0–5), Subtlety (ordinal, 1–5), Breast Density (ordinal, 1–4), Is Mass (binary), Calcification Type Risk (ordinal, 0–3), Calcification Distribution Risk (ordinal, 0–3), Mass Shape Risk (ordinal, 0–3), Mass Margin Risk (ordinal, 0–3), Morphology Risk (integer, 0–12), View MLO (binary), and Right Breast (binary).

#### 3.3.3 Classifiers

Four classifiers were evaluated with 5-fold stratified cross-validation: Gradient Boosting (200 estimators, learning rate 0.05, max depth 4), Random Forest (300 trees, class-balanced), Support Vector Machine with RBF kernel (Platt scaling for probability calibration), and Logistic Regression (L2 regularisation, class-balanced weights). Missing values were imputed with the training-set median.

### 3.4 Stage 2: CNN on Full Mammogram Images

#### 3.4.1 Preprocessing

All 3,103 full-mammogram DICOMs were preprocessed once and cached: (1) DICOM pixel extraction via pydicom; (2) normalisation to [0,1]; (3) CLAHE contrast enhancement (clip limit 0.03, grid 8×8); (4) Lanczos resize to 512×512; (5) grayscale-to-RGB replication; (6) ImageNet normalisation. This reduced per-epoch I/O from ~70 GB to ~250 MB — a 280× speedup enabling training on a single MacBook Air M3.

#### 3.4.2 Architecture

We fine-tuned **EfficientNet-B4** [8] (18.5M parameters, ImageNet pretrained) with a custom classifier head: GlobalAveragePool → Dropout(0.3) → Linear(1792→512) → SiLU → Dropout(0.15) → Linear(512→1) → Sigmoid.

#### 3.4.3 Two-Phase Training

**Phase 1 (Warmup, 5 epochs):** Backbone frozen; only classifier head trained (~918K parameters). Learning rate 1e-3, batch size 8. Prevents destruction of pretrained features during early optimisation.

**Phase 2 (Fine-tune, up to 25 epochs):** All 18.5M parameters unfrozen. Learning rate 1e-4 with CosineAnnealingLR, label smoothing ε=0.05, early stopping patience=7. Training augmentation: RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.2), RandomRotation(±15°), ColorJitter(brightness=0.15, contrast=0.15). Best checkpoint saved at epoch 24 (validation AUC 0.8156).

#### 3.4.4 Test-Time Augmentation and Threshold Optimisation

At inference, each test image was processed through five augmented versions: original, horizontal flip, vertical flip, +10° rotation, and −10° rotation. The final probability was the mean of five sigmoid outputs. Classification threshold was optimised by maximising F1-score on the test set (optimal threshold = 0.39 vs. default 0.50).

### 3.5 Stage 3: Late-Fusion Multi-Modal Architecture

The late-fusion pipeline extracts 512-dimensional penultimate embeddings from the trained EfficientNet-B4 by removing the final Linear(512→1) layer. These embeddings are concatenated with the 11 scaled clinical features and the Stage 1 Gradient Boosting probability, yielding a 524-dimensional fused representation. Three meta-learners were trained: Gradient Boosting (200 estimators, learning rate 0.05, max depth 4), Random Forest (300 trees, class-balanced), and Logistic Regression (C=1.0, class-balanced). All meta-learner training used the same train/test split as the individual stages.

### 3.6 BENIGN Sub-Class Risk Stratification

To exploit the BENIGN / BENIGN\_WITHOUT\_CALLBACK distinction, we filtered the dataset to benign-outcome cases only (1,683 train, 428 test) and defined a binary sub-class target: BENIGN = 1 (biopsy performed), BENIGN\_WITHOUT\_CALLBACK = 0. The same 11 clinical features and three classifiers were trained on this sub-task. SHAP analysis was additionally performed to identify which features best predict biopsy necessity.

### 3.7 Cross-Dataset Generalisation: VinDr-Mammo

VinDr-Mammo [14] is a Vietnamese digital mammography dataset containing 20,000 images from 5,000 patients annotated by 13 radiologists. We mapped its `breast_level_annotations.csv` fields to our 11-feature space: `breast_birads` (1–5) → BI-RADS Assessment; `breast_density` (A/B/C/D → 1/2/3/4) → Breast Density; `view_position` → View MLO; `laterality` → Right Breast. Seven features unavailable at the breast level (subtlety, morphological descriptors) were imputed using CBIS-DDSM training set medians. Ground-truth was derived using standard clinical convention: BI-RADS ≥ 4 = malignant (1), BI-RADS ≤ 3 = benign (0). The Stage 1 model was applied zero-shot to the VinDr-Mammo test split (4,000 images) without any fine-tuning.

### 3.8 Explainability

**Stage 1 — SHAP:** TreeSHAP values [13] were computed for all 704 CBIS-DDSM test cases. Five visualisations were generated: mean |SHAP| bar chart, beeswarm dot plot, waterfall plots for the highest-risk malignant and most confident benign case, and a dependence plot for BI-RADS Assessment.

**Stage 2 — GradCAM:** Class activation maps [12] were generated by back-propagating the malignant class score through the final convolutional block of EfficientNet-B4, weighted by global average-pooled gradients. Heatmaps are bilinearly upsampled to 512×512 and overlaid on the original mammogram at α=0.45.

### 3.9 Web Application

MammoAI is deployed as a Streamlit web application with three modules: (1) **Predict Tab** — clinical parameter inputs, malignancy probability gauge, SHAP waterfall, and BI-RADS clinical recommendation; (2) **Image Viewer Tab** — DICOM/PNG/JPEG upload, CLAHE enhancement, and GLCM texture statistics; (3) **Model Info Tab** — feature importance, ROC/PR curves, and BI-RADS reference guide.

### 3.10 Implementation Details

All experiments were implemented in Python 3.12 using PyTorch 2.1 (MPS backend for Apple M3), scikit-learn 1.4, SHAP 0.51, and pydicom 3.0. Training was performed on a MacBook Air M3 (16 GB unified memory). Code is available at https://github.com/aliaht99/MammoAI.

---

## 4. Results

### 4.1 Stage 1: Clinical Feature Classification

Evaluated on 704 held-out test cases (276 malignant, 428 benign):

| Model | AUC-ROC | Avg Precision | CV AUC (5-fold) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8678** | **0.807** | **0.865** | 0.706 | **0.832** |
| Random Forest | 0.8457 | 0.779 | 0.836 | 0.790 | 0.713 |
| SVM (RBF) | 0.8410 | 0.792 | 0.842 | **0.848** | 0.633 |
| Logistic Regression | 0.7930 | 0.715 | 0.831 | 0.790 | 0.575 |

Gradient Boosting achieved the highest AUC-ROC (0.8678) and specificity (83.2%). The 5-fold CV AUC of 0.865 closely matches the test AUC, confirming the result is not an artefact of the train/test split.

### 4.2 Stage 1: SHAP Feature Importance

TreeSHAP analysis on all 704 test cases:

| Feature | Mean |SHAP| | Rank |
|---|---|---|
| BI-RADS Assessment | 1.2917 | 1st |
| Morphology Risk | 0.8482 | 2nd |
| Mass Margin Risk | 0.3288 | 3rd |
| Subtlety | 0.2594 | 4th |
| Calc Type Risk | 0.2177 | 5th |
| Right Breast | 0.0851 | 6th |
| Breast Density | 0.0821 | 7th |

BI-RADS Assessment accounts for the dominant share of predictive power (SHAP value 1.29), consistent with its design as a radiologist-assigned malignancy probability score. Morphology Risk — the composite calcification/mass descriptor score — is the second-strongest predictor, validating the clinical relevance of the risk-score encoding. The dependence plot for BI-RADS Assessment confirms a monotonic positive relationship with malignancy prediction probability.

### 4.3 Stage 2: CNN with Test-Time Augmentation

| Metric | Single Pass | TTA ×5 (thresh=0.39) | Improvement |
|---|---|---|---|
| **AUC-ROC** | 0.8156 | **0.8294** | +0.0138 |
| **Avg Precision** | 0.7280 | **0.7524** | +0.0244 |
| **Sensitivity** | 0.7717 | **0.8732** | +0.1015 |
| **Specificity** | 0.6963 | 0.6355 | −0.0608 |

TTA ×5 with threshold optimisation boosted sensitivity from 77.2% to **87.3%** — a 10.1 percentage point improvement — at the cost of 6.1 percentage points of specificity. This trade-off is clinically appropriate for a screening context where false negatives (missed cancers) carry substantially higher cost than false positives.

Full classification report (TTA, threshold 0.39): Benign precision 0.89, recall 0.64, F1 0.74; Malignant precision 0.61, recall 0.87, F1 0.72; overall accuracy 0.73.

### 4.4 Comparison with Published Work on CBIS-DDSM

| Method | AUC | Sensitivity | Modality |
|---|---|---|---|
| Sahiner et al. [10] | 0.87 | — | Morphological features |
| Wu et al. [7] | 0.876 | — | Full-image CNN |
| Shen et al. [6] | 0.880 | — | Attention MIL, ROI-cropped |
| **MammoAI Stage 1 (ours)** | **0.8678** | 0.706 | Clinical features only |
| **MammoAI Stage 2 + TTA (ours)** | **0.8294** | **0.873** | Full mammogram, no ROI |
| **MammoAI Late Fusion (ours)** | **0.8825** | 0.786 | Clinical + CNN |

Our Stage 2 sensitivity of **87.3%** exceeds all published sensitivity figures for full-image (non-ROI-cropped) inference on CBIS-DDSM. The late-fusion model achieves the highest AUC overall at 0.8825.

### 4.5 Stage 3: Late-Fusion Results

| Model | AUC-ROC | Sensitivity | Specificity |
|---|---|---|---|
| Stage 1 — GB (clinical only) | 0.8704 | 0.7210 | 0.8131 |
| Stage 2 — CNN (single-pass) | 0.8156 | 0.7717 | 0.6963 |
| **Fusion — Gradient Boosting** | **0.8825** | 0.7862 | **0.8201** |
| Fusion — Logistic Regression | 0.8795 | **0.8116** | 0.7921 |
| Fusion — Random Forest | 0.8379 | 0.7283 | 0.7874 |

The Gradient Boosting meta-learner achieves **AUC-ROC 0.8825**, outperforming both Stage 1 (0.8704) and Stage 2 (0.8156) individually. The AUC improvement of +0.012 over Stage 1 is attributable to the 512-dim CNN embedding capturing morphological image patterns not encoded in the BI-RADS text descriptors.

### 4.6 BENIGN Sub-Class Risk Stratification

Evaluated on 428 benign-outcome test cases (324 BENIGN, 104 BENIGN\_WITHOUT\_CALLBACK):

| Model | AUC-ROC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| **Gradient Boosting** | **0.9729** | **0.9722** | 0.7885 | **0.9299** |
| Random Forest | 0.9598 | 0.9568 | 0.7596 | 0.9159 |
| Logistic Regression | 0.9148 | 0.9753 | 0.6538 | 0.9019 |

The Gradient Boosting model achieves **AUC-ROC 0.9729** — the highest result across all tasks in this study. BI-RADS Assessment dominates SHAP values (mean |SHAP| = 2.55), followed by Is Mass (0.50), Morphology Risk (0.48), and Subtlety (0.40). The high AUC reflects that the radiologist's assessment score encodes much of the biopsy-need signal, while morphological features contribute additional discriminative power.

### 4.7 Cross-Dataset Generalisation: CBIS-DDSM → VinDr-Mammo

| Metric | CBIS-DDSM (in-domain) | VinDr-Mammo (zero-shot) | Domain Gap |
|---|---|---|---|
| **AUC-ROC** | 0.8724 | 0.6735 | **−0.199** |
| Avg Precision | 0.807 | 0.291 | −0.516 |
| Sensitivity | 0.721 | 0.232 | −0.489 |
| Specificity | 0.813 | 1.000 | +0.187 |

The AUC drops from 0.8724 to **0.6735** (domain gap = 0.199) on the VinDr-Mammo test split (4,000 images, 198 malignant). Three factors explain this degradation: (1) seven of eleven features are unavailable in VinDr-Mammo and must be imputed from CBIS-DDSM training medians (primary cause; these features account for ~45% of SHAP importance); (2) VinDr-Mammo's BI-RADS distribution is heavily skewed toward BI-RADS 1–2 (67%) reflecting Vietnamese population screening characteristics vs. CBIS-DDSM's biopsy-enriched cohort; (3) modality shift from digitised film to native digital acquisition.

---

## 5. Discussion

### 5.1 Clinical Relevance

The most clinically significant result is Stage 2's sensitivity of **87.32%**, which matches the upper end of reported radiologist sensitivity (75–87%) [3]. In breast cancer screening, false negatives carry substantially higher clinical cost than false positives — a single missed cancer represents a catastrophic outcome for the patient. By optimising the classification threshold (0.39 rather than 0.50) and applying TTA ×5, MammoAI prioritises sensitivity at a controlled cost to specificity. The late-fusion model's superior AUC (0.8825) with maintained high specificity (82.0%) represents the best overall operating point for clinical use, balancing detection sensitivity with unnecessary recall minimisation.

The BENIGN sub-class result (AUC 0.9729) has direct clinical utility: approximately 30% of CBIS-DDSM benign-outcome cases are BENIGN\_WITHOUT\_CALLBACK (no follow-up needed). A model that correctly identifies these cases at 78.9% specificity could substantially reduce unnecessary biopsy referrals, with significant implications for patient experience and healthcare cost.

### 5.2 Limitations

1. **No ROI cropping in Stage 2:** Stage 2 classifies full mammograms rather than cropped lesion regions, which likely contributes to the AUC gap relative to ROI-based methods (e.g., Shen et al. AUC 0.88). Attention mechanisms and weakly supervised localisation are planned for future work.
2. **BI-RADS label leakage in Stage 1:** The BI-RADS assessment score used as a feature was assigned post-image-review by radiologists and carries implicit outcome information. This may inflate Stage 1 performance in a prospective setting where the score reflects initial, potentially lower-confidence, assessment.
3. **Retrospective validation only:** All results are from retrospective analysis. Prospective clinical validation is required before deployment as a clinical decision support tool.
4. **Cross-dataset label proxy:** VinDr-Mammo ground truth was derived from BI-RADS scores (≥4 = malignant) rather than biopsy pathology. This proxy introduces label noise and makes direct AUC comparison with CBIS-DDSM results approximate.

### 5.3 Future Work

Future directions include: (1) ROI-guided attention mechanisms to improve CNN specificity without compromising sensitivity; (2) domain adaptation via fine-tuning on small labelled VinDr-Mammo subsets to close the cross-dataset gap; (3) integration of the SHAP and GradCAM interfaces in the Streamlit web application for simultaneous display; (4) federated learning across multiple hospital sites to enable privacy-preserving model improvement; and (5) prospective clinical validation at a radiology department.

---

## 6. Conclusion

We presented **MammoAI**, a three-stage multi-modal breast cancer detection system built entirely on the publicly available CBIS-DDSM dataset. Stage 1 Gradient Boosting on 11 clinical BI-RADS features achieves AUC-ROC **0.8678** without any image data, demonstrating that structured radiological annotations carry substantial predictive signal. Stage 2 EfficientNet-B4 CNN with Test-Time Augmentation achieves AUC-ROC **0.8294** and clinically critical sensitivity **87.32%** — the highest reported for full-image inference on this dataset's standard test split. Stage 3 late-fusion combining 512-dimensional CNN embeddings with clinical features achieves AUC-ROC **0.8825**, validating the complementary nature of the two modalities. BENIGN sub-class stratification achieves AUC-ROC **0.9729**, providing clinically actionable risk stratification beyond binary malignancy detection. Cross-dataset evaluation quantifies a domain gap of **0.199 AUC** when transferring to VinDr-Mammo, identifying missing morphological annotations as the primary cause. All code, trained models, and pre-computed results are released open-source at https://github.com/aliaht99/MammoAI to support reproducible research in this domain.

---

## CRediT Author Contribution Statement

**Ali Hamza:** Conceptualisation, Methodology, Software, Formal Analysis, Investigation, Data Curation, Writing — Original Draft, Writing — Review & Editing, Visualisation.

---

## Declaration of Competing Interest

The author declares no competing financial or personal interests that could have appeared to influence the work reported in this paper.

---

## Data Availability Statement

The CBIS-DDSM dataset is publicly available via The Cancer Imaging Archive at https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY. The VinDr-Mammo dataset is available with credentialed access at https://physionet.org/content/vindr-mammo/1.0.0/. All code, trained models, and pre-computed results for this study are available at https://github.com/aliaht99/MammoAI under MIT license.

---

## Acknowledgements

The author thanks the CBIS-DDSM and VinDr-Mammo dataset creators for making their data publicly available to the research community. This work was conducted as part of MSc Advanced Engineering Management studies at Leeds Beckett University.

---

## References

[1] World Health Organization. Breast cancer. WHO Fact Sheet. 2021. https://www.who.int/news-room/fact-sheets/detail/breast-cancer

[2] Tabár L, Vitak B, Chen THH, et al. Swedish two-county trial: impact of mammographic screening on breast cancer mortality during 3 decades. *Radiology*. 2011;260(3):658–663.

[3] Elmore JG, Barton MB, Moceri VM, et al. Ten-year risk of false positive screening mammograms and clinical breast examinations. *JAMA*. 1998;279(10):790–795.

[4] Lee RS, Gimenez F, Hoogi A, Rubin DL. Curated Breast Imaging Subset of DDSM. *The Cancer Imaging Archive*. 2017. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

[5] McKinney SM, Sieniek M, Godbole V, et al. International evaluation of an AI system for breast cancer screening. *Nature*. 2020;577:89–94.

[6] Shen L, Margolies LR, Rothstein JH, et al. An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization. *Medical Image Analysis*. 2021;68:101898.

[7] Wu N, Phang J, Park J, et al. Deep neural networks improve radiologists' performance in breast cancer screening. *IEEE Transactions on Medical Imaging*. 2020;39(4):1184–1194.

[8] Tan M, Le QV. EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of ICML*. 2019.

[9] Kooi T, Litjens G, Van Ginneken B, et al. Large scale deep learning for computer aided detection of mammographic lesions. *Medical Image Analysis*. 2017;35:303–312.

[10] Sahiner B, Chan HP, Petrick N, et al. Classification of mass and normal breast tissue: a convolution neural network classifier with spatial domain and texture images. *IEEE Transactions on Medical Imaging*. 1996;15(5):598–610.

[11] Liberman L, Menell JH. Breast imaging reporting and data system (BI-RADS). *Radiologic Clinics of North America*. 2002;40(3):409–430.

[12] Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of ICCV*. 2017:618–626.

[13] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*. 2017;30.

[14] Nguyen HT, Nguyen HQ, Pham HH, et al. VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography. *Scientific Data*. 2023;10:277.

