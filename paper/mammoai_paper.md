# MammoAI: Early Breast Cancer Detection Using Clinical Radiological Features and Deep Learning on the CBIS-DDSM Mammography Dataset

**Ali Hamza**  
[Your Institution / Affiliation]  
[Your Email]  
[Date: 2026]

---

## Abstract

Breast cancer remains one of the leading causes of cancer-related mortality among women worldwide. Early and accurate detection through mammography screening is critical for improving survival outcomes. In this paper, we present **MammoAI**, a two-stage computer-aided detection (CAD) system trained on the publicly available **CBIS-DDSM** (Curated Breast Imaging Subset of the Digital Database for Screening Mammography) dataset comprising 3,568 annotated mammography cases. In the first stage, we engineer clinical and radiological features — including BI-RADS assessment scores, breast density, calcification morphology, and mass characteristics — and evaluate four classical machine learning classifiers. Our best Stage 1 model (Gradient Boosting) achieves an **AUC-ROC of 0.868**, sensitivity of 0.706, and specificity of 0.832 on the held-out test set. In the second stage (in progress), we fine-tune an **EfficientNet-B4** convolutional neural network directly on the 10,239 raw DICOM mammogram images using transfer learning and Apple Silicon MPS acceleration. MammoAI is released as an open-source, interactive Streamlit web application enabling clinicians and researchers to assess malignancy risk from either clinical parameters or uploaded mammogram images.

**Keywords:** breast cancer detection, mammography, CBIS-DDSM, machine learning, deep learning, EfficientNet, BI-RADS, computer-aided detection, transfer learning

---

## 1. Introduction

Breast cancer is the most frequently diagnosed cancer among women globally, accounting for approximately 2.3 million new cases and 685,000 deaths in 2020 alone [WHO, 2021]. Mammography remains the gold-standard screening modality, with population-level programs demonstrated to reduce breast cancer mortality by 20–40% [Tabár et al., 2003]. However, interpretation of mammograms is subjective, time-consuming, and subject to inter-reader variability, with false-negative rates ranging from 10–30% [Elmore et al., 1998].

Computer-aided detection (CAD) systems have emerged as a promising supplement to radiologist interpretation. Early CAD systems relied on handcrafted features derived from radiological reports — including the Breast Imaging Reporting and Data System (BI-RADS) lexicon — and classical machine learning classifiers. More recently, deep convolutional neural networks (CNNs) trained on large mammography corpora have demonstrated radiologist-level performance [McKinney et al., 2020; Shen et al., 2021].

Despite these advances, several challenges remain:
1. Most high-performing models are trained on proprietary datasets unavailable to the research community.
2. Existing open-source implementations rarely provide end-to-end pipelines from raw DICOM to clinical prediction.
3. Few systems simultaneously leverage both structured clinical annotations and raw pixel features.

This paper addresses these gaps with the following contributions:

- **A fully reproducible ML pipeline** built entirely on the publicly available CBIS-DDSM dataset.
- **A structured clinical feature extraction framework** mapping BI-RADS descriptors to quantitative risk scores.
- **A comparative evaluation** of four classical classifiers (Logistic Regression, Random Forest, Gradient Boosting, SVM).
- **An open-source Streamlit web application** enabling real-time malignancy risk assessment.
- **A transfer learning CNN pipeline** (Stage 2) fine-tuned on 10,239 DICOM mammograms using Apple M3 MPS acceleration.

---

## 2. Related Work

### 2.1 Classical Machine Learning Approaches

Early CAD systems for mammography used decision trees, SVMs, and neural networks on handcrafted features including shape, margin, and texture descriptors extracted from segmented regions of interest [Eltoukhy et al., 2010]. Sahiner et al. [1996] demonstrated that linear discriminant analysis on morphological features could distinguish malignant from benign masses with AUC ≈ 0.87. Subsequent work incorporated BI-RADS lexicon features directly, showing that assessment scores alone carry substantial predictive signal [Liberman et al., 2002].

### 2.2 Deep Learning Approaches

The advent of large-scale mammography datasets enabled training of deep CNNs. Kooi et al. [2017] trained a CNN on 45,000 mammograms achieving AUC 0.93. Wu et al. [2019] proposed a globally-aware approach using both patch-level and image-level representations, reaching radiologist-level performance on CBIS-DDSM. McKinney et al. [2020] demonstrated superhuman performance using a proprietary dataset of over 28,000 women.

### 2.3 CBIS-DDSM Benchmarks

Lee et al. [2017] curated and released the CBIS-DDSM dataset, providing standardized train/test splits to enable reproducible benchmarking. Prior work on this dataset reports AUC ranging from 0.76 (classical ML) to 0.88 (deep CNNs). Our work establishes a new reproducible baseline with clear feature engineering methodology.

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
- DICOM file paths for full mammograms, cropped ROIs, and binary masks

### 3.2 Data Access

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
| COARSE / EGGSHELL | 0 | Benign entities |

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

1. **Logistic Regression** — L2 regularization, class-balanced weights, max_iter=1000
2. **Random Forest** — 300 trees, class-balanced, max_features=sqrt
3. **Gradient Boosting** — 200 estimators, learning_rate=0.05, max_depth=4
4. **SVM (RBF kernel)** — class-balanced, probability calibration via Platt scaling

Missing values were imputed with the training-set median. All models were evaluated with 5-fold stratified cross-validation.

### 4.3 Stage 2: CNN on DICOM Images (In Progress)

For Stage 2, we fine-tune **EfficientNet-B4** [Tan & Le, 2019] pre-trained on ImageNet on the 10,239 DICOM mammogram images:

**Preprocessing pipeline:**
1. Read DICOM via pydicom, extract pixel array
2. Apply CLAHE (adaptive histogram equalization) for contrast normalization
3. Resize to 512×512
4. Normalize to ImageNet mean/std
5. Augmentation: random horizontal flip, ±15° rotation, ±10% brightness

**Fine-tuning strategy:**
- Freeze all layers except the classifier head for 5 epochs (warmup)
- Unfreeze all layers, fine-tune at 1/10 learning rate for 20 epochs
- Loss: Binary Cross-Entropy with label smoothing (ε=0.1)
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Hardware: Apple MacBook Air M3 (MPS backend)

---

## 5. Experimental Results

### 5.1 Stage 1 Results

| Model | AUC-ROC | Avg Precision | CV AUC (5-fold) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.868** | **0.807** | **0.865** | 0.706 | **0.832** |
| Random Forest | 0.846 | 0.779 | 0.836 | 0.790 | 0.713 |
| SVM (RBF) | 0.841 | 0.792 | 0.842 | **0.848** | 0.633 |
| Logistic Regression | 0.793 | 0.715 | 0.831 | 0.790 | 0.575 |

Gradient Boosting achieved the highest AUC-ROC (0.868) and specificity (0.832), while SVM achieved the highest sensitivity (0.848). For screening applications where minimizing false negatives is critical, an ensemble of Gradient Boosting and SVM may offer the best clinical trade-off.

### 5.2 Feature Importance

The three most informative features in the Gradient Boosting model were:
1. **BI-RADS Assessment** (importance: ~0.42) — the single strongest predictor
2. **Morphology Risk** (importance: ~0.18) — composite calcification/mass descriptor score
3. **Subtlety** (importance: ~0.12) — radiologist-perceived obviousness

This aligns with clinical understanding: BI-RADS was specifically designed to encode malignancy suspicion, while morphological descriptors (spiculation, pleomorphic calcifications) have established pathological correlates.

### 5.3 Class Imbalance Analysis

The training set contains 1,181 malignant (41.2%) and 1,683 benign (58.8%) cases — a moderate imbalance addressed through class-balanced loss weighting in all models.

---

## 6. Web Application

MammoAI is deployed as a Streamlit web application with three functional modules:

1. **Predict Tab:** Users input clinical parameters via dropdown menus and sliders. The Gradient Boosting model returns a malignancy probability, risk tier (LOW / MEDIUM / HIGH / VERY HIGH), and a clinical recommendation aligned with ACR BI-RADS follow-up guidelines.

2. **Image Viewer Tab:** Accepts DICOM, PNG, or JPEG uploads. Displays original and enhanced mammogram side-by-side with adjustable contrast/brightness/CLAHE controls. Extracts and displays image texture statistics (GLCM contrast, energy, homogeneity, correlation).

3. **Model Info Tab:** Displays feature importance, ROC/PR curves, model comparison table, and a BI-RADS reference guide.

---

## 7. Discussion

### 7.1 Clinical Relevance

Our Stage 1 AUC of 0.868 is competitive with prior classical ML work on CBIS-DDSM (typically 0.76–0.85) and approaches the lower bound of CNN-based systems. Critically, our model is interpretable — feature importances directly map to clinical decision criteria — making it suitable for deployment in resource-limited settings where radiologist expertise is scarce.

### 7.2 Limitations

1. **Stage 1 only uses metadata:** The 10,239 DICOM images (152 GB) have not yet been incorporated. Stage 2 CNN training is expected to substantially improve performance.
2. **Single dataset:** Generalizability to other scanner vendors, patient demographics, or imaging protocols is not yet established.
3. **No prospective validation:** All results are retrospective; prospective clinical validation is required before clinical deployment.
4. **BI-RADS leakage risk:** Because BI-RADS assessment was assigned by a radiologist who had already viewed the image, it carries implicit information about the outcome. Real-world deployment should be tested in a prospective pipeline.

### 7.3 Future Work

- **Stage 2 CNN:** Fine-tune EfficientNet-B4 on the full 10,239-image corpus using Apple M3 MPS acceleration.
- **Multi-modal fusion:** Combine CNN image features with clinical features in a late-fusion architecture.
- **Explainability:** Add Grad-CAM saliency maps to the Streamlit app to highlight suspicious regions.
- **External validation:** Evaluate on VinDr-Mammo or INbreast datasets.
- **Federated learning:** Explore privacy-preserving training across multiple hospital sites.

---

## 8. Conclusion

We presented MammoAI, an open-source breast cancer detection system built on the CBIS-DDSM dataset. Our Gradient Boosting model trained on clinical radiological features achieves AUC-ROC 0.868 without using any raw image data, demonstrating that structured BI-RADS annotations carry substantial predictive signal. The accompanying Streamlit application provides an accessible interface for clinicians and researchers. Stage 2 CNN training on the full 152 GB DICOM corpus is underway and expected to further improve performance. All code, model weights, and pre-computed results are released at: https://github.com/YOUR_USERNAME/MammoAI

---

## References

1. WHO (2021). Breast cancer. World Health Organization. https://www.who.int/news-room/fact-sheets/detail/breast-cancer
2. Tabár, L., et al. (2003). The Swedish two-county trial twenty years later. *Radiology*, 228(1), 263–268.
3. Elmore, J. G., et al. (1998). Ten-year risk of false positive screening mammograms. *JAMA*, 279(10), 790–795.
4. Lee, R. S., Gimenez, F., Hoogi, A., & Rubin, D. (2017). Curated Breast Imaging Subset of DDSM. *The Cancer Imaging Archive*. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
5. McKinney, S. M., et al. (2020). International evaluation of an AI system for breast cancer screening. *Nature*, 577, 89–94.
6. Shen, L., et al. (2021). An interpretable classifier for high-resolution breast cancer screening images. *Medical Image Analysis*, 68, 101898.
7. Wu, N., et al. (2019). Deep neural networks improve radiologists' performance in breast cancer screening. *IEEE TMI*, 39(4), 1184–1194.
8. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for CNNs. *ICML 2019*.
9. Kooi, T., et al. (2017). Large scale deep learning for computer aided detection of mammographic lesions. *Medical Image Analysis*, 35, 303–312.
10. Sahiner, B., et al. (1996). Classification of mass and normal breast tissue. *IEEE TMI*, 15(5), 598–610.
11. Liberman, L., & Menell, J. H. (2002). Breast imaging reporting and data system (BI-RADS). *Radiologic Clinics*, 40(3), 409–430.

---

*Corresponding author: Ali Hamza — [your.email@institution.edu]*  
*Code & data: https://github.com/YOUR_USERNAME/MammoAI*  
*License: MIT*
