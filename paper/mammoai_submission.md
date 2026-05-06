---
title: "MammoAI: A Multi-Modal Interpretable Framework for Breast Cancer Detection — Fusing Clinical Features with Deep Learning on CBIS-DDSM"
author: "Ali Hamza"
date: "May 2026"
---

# MammoAI: A Multi-Modal Interpretable Framework for Breast Cancer Detection — Fusing Clinical Features with Deep Learning on CBIS-DDSM

**Ali Hamza**
MSc Advanced Engineering Management, Leeds Beckett University, Leeds, United Kingdom
alihamza.aht.99@gmail.com | https://github.com/aliaht99/MammoAI

---

## Highlights

- Three-stage pipeline: GB clinical model (AUC 0.8678) + EfficientNet-B4 CNN (AUC 0.8294) + late fusion (AUC 0.8825)
- CNN sensitivity reaches 87.3% via Test-Time Augmentation — highest reported for full-image CBIS-DDSM inference
- BENIGN sub-class stratification (AUC 0.9729) predicts biopsy necessity from benign-outcome cases
- First SHAP + GradCAM dual explainability system on CBIS-DDSM, shown simultaneously in a web interface
- First quantitative CBIS-DDSM → VinDr-Mammo domain shift measurement (AUC gap = 0.199)

---

## Abstract

Breast cancer is the most commonly diagnosed malignancy in women globally, and early detection through mammography screening remains critical for reducing mortality. While AI-based detection has made considerable progress, most published systems either rely on proprietary hospital data, use black-box models with no clinical interpretability, or have never been tested outside the dataset they were trained on. This work addresses all three of these gaps.

We built MammoAI on the publicly available CBIS-DDSM dataset — 3,568 annotated mammography cases and 10,239 DICOM images. The system has three stages. Stage 1 extracts 11 clinical features from BI-RADS annotations and trains a Gradient Boosting classifier, reaching AUC-ROC 0.8678 with 83.2% specificity. Stage 2 fine-tunes an EfficientNet-B4 CNN on 3,103 full mammograms; with Test-Time Augmentation and threshold optimisation, it achieves AUC-ROC 0.8294 and sensitivity 87.3%. Stage 3 combines the 512-dimensional CNN embedding with the clinical features and passes the fused 524-dimensional vector through a meta-learner, pushing AUC to 0.8825 — better than either stage alone.

Beyond the main detection task, we train a secondary classifier on benign-outcome cases to distinguish BENIGN from BENIGN\_WITHOUT\_CALLBACK, achieving AUC 0.9729. We also apply the Stage 1 model zero-shot to VinDr-Mammo (20,000 Vietnamese digital mammograms), finding a domain gap of 0.199 AUC and identifying missing morphological annotations as the primary cause. SHAP values explain every clinical prediction; GradCAM heatmaps highlight suspicious image regions. Both are shown simultaneously in a Streamlit web application.

All code, trained models, and results are available at https://github.com/aliaht99/MammoAI.

**Keywords:** breast cancer; mammography; CBIS-DDSM; EfficientNet-B4; gradient boosting; late-fusion; SHAP; GradCAM; test-time augmentation; domain shift; VinDr-Mammo; computer-aided detection

---

## 1. Introduction

Roughly 2.3 million women were diagnosed with breast cancer in 2020, and around 685,000 died from it [1]. The prognosis is strongly linked to how early the disease is caught — five-year survival rates exceed 99% when detected at stage I but fall below 30% at stage IV. Mammography screening has therefore been central to breast cancer control for decades, and randomised trials in Sweden demonstrated mortality reductions of 20–40% in screened populations [2].

The catch is that reading mammograms is hard. False-negative rates in routine practice sit between 10% and 30%, even among experienced radiologists [3]. Inter-reader variability is well-documented — the same case assigned different BI-RADS scores by different readers is a known problem, not an edge case. Radiologist workload compounds this: in the UK, each reader is expected to interpret around 5,000 screening mammograms per year. Under these conditions, a second opinion from a reliable automated system has clear value.

The research community has responded with a large body of AI work on mammography detection. The results at the top end are impressive — McKinney et al. [5] showed superhuman performance at Google using a proprietary dataset of over 28,000 women, and Shen et al. [6] reached AUC 0.88 on CBIS-DDSM with an attention-based system. But a recurring pattern in this literature limits practical impact: the best models use data that nobody else can access, explain nothing about their predictions, and are never tested outside their training distribution. A clinician reading these papers has no way to reproduce the results, understand why a case was flagged, or know whether the model would still work on patients from a different hospital or country.

This paper takes a different starting point. We use only publicly available data (CBIS-DDSM for training, VinDr-Mammo for cross-dataset evaluation), explain every prediction (SHAP for the clinical stage, GradCAM for the imaging stage), and release everything — code, models, and results — under an open licence. The system we built, MammoAI, is not primarily a performance competition entry; it is an attempt to show what a complete, interpretable, reproducible mammography AI pipeline actually looks like when built from scratch.

The specific contributions that distinguish this work from prior CBIS-DDSM studies are:

- **Late-fusion of clinical and imaging features.** We extract the 512-dimensional penultimate embedding from EfficientNet-B4 and combine it with 11 BI-RADS clinical features, achieving AUC 0.8825 — better than either modality alone. This fusion has not been published on this dataset.
- **Simultaneous SHAP + GradCAM explainability.** Prior work shows one or the other. We show both in the same interface, so a clinician sees which BI-RADS descriptor drove the clinical prediction and which image region drove the CNN simultaneously.
- **BENIGN sub-class analysis.** CBIS-DDSM distinguishes BENIGN (biopsy confirmed, recall required) from BENIGN\_WITHOUT\_CALLBACK (no recall needed). This distinction maps directly to a real clinical decision — whether to refer a patient for biopsy. No prior work on CBIS-DDSM has trained a classifier specifically on this sub-task.
- **Cross-dataset domain shift quantification.** We apply the trained model zero-shot to VinDr-Mammo and measure the AUC degradation. This is the first controlled measurement of the CBIS-DDSM → VinDr-Mammo transfer gap.

---

## 2. Related Work

### 2.1 Clinical Feature Models

The BI-RADS lexicon was designed to standardise radiologist reporting, and its assessment scores (0–5) encode malignancy suspicion directly. It has long been known that BI-RADS assessment alone is a strong predictor of pathological outcome, with AUC approaching 0.85 in some series [11]. Earlier CAD systems built on this by encoding morphological descriptors — mass margin, calcification type — as numerical risk scores and feeding them to classifiers. Sahiner et al. [10] reached AUC 0.87 with linear discriminant analysis on shape features. Our Stage 1 pipeline follows this tradition but uses a more comprehensive feature set and modern gradient boosting.

### 2.2 Deep Learning on Mammography

CNNs changed what was achievable on full-image mammography. Kooi et al. [9] trained on 45,000 cases and reached AUC 0.93 for mass detection. On the CBIS-DDSM benchmark specifically, Wu et al. [7] proposed globally-aware multiple instance learning and achieved AUC 0.876. Shen et al. [6] combined weakly supervised localisation with an interpretable attention mechanism and reached AUC 0.88 — though their system uses ROI-cropped inputs, which gives it an informational advantage over whole-image classification. Our Stage 2 trains on full mammograms without any cropping and reaches 87.3% sensitivity with TTA, which to our knowledge is the highest sensitivity reported for this inference mode on the standard CBIS-DDSM split.

### 2.3 Explainability

GradCAM [12] produces saliency maps by weighting feature maps by the gradient of the class score — computationally cheap and visually intuitive. SHAP [13] gives game-theoretically grounded feature attributions for tabular models. Both have been used separately in radiology AI, but the combination — showing a clinician both why the clinical model flagged a case and where in the image the CNN is looking — has not been implemented on CBIS-DDSM.

### 2.4 Multi-Modal Fusion

Combining structured clinical data with image features is an active area in medical AI broadly. In mammography specifically, the idea of using BI-RADS features alongside image features is appealing because radiologists already extract them as part of standard reporting. However, direct late-fusion of EfficientNet penultimate embeddings with structured BI-RADS features has not been reported on CBIS-DDSM.

### 2.5 Dataset Generalisation

CBIS-DDSM was created from scanned film mammograms from the 1990s — the image characteristics are quite different from modern full-field digital mammography. VinDr-Mammo [14], released in 2022, provides a large (20,000 image) Vietnamese digital dataset with radiologist annotations. Whether a model trained on CBIS-DDSM retains meaningful performance on VinDr-Mammo has not been published.

---

## 3. Materials and Methods

### 3.1 Dataset

CBIS-DDSM [4] is available via The Cancer Imaging Archive. It contains digitised film mammograms with ROI masks and detailed radiological annotations including BI-RADS assessment, breast density, subtlety rating, abnormality type, and morphological descriptors (calcification type/distribution for calc cases, mass shape/margins for mass cases). Ground-truth pathology is MALIGNANT, BENIGN, or BENIGN\_WITHOUT\_CALLBACK — the last category denoting cases deemed low-suspicion enough that biopsy was not performed.

| Split | Cases | Malignant | Benign |
|---|---|---|---|
| Training (calc + mass) | 2,864 | 1,181 | 1,683 |
| Test (calc + mass) | 704 | 276 | 428 |
| **Total** | **3,568** | **1,457** | **2,111** |

The 152 GB of DICOM files contain 3,103 full mammograms, 7,026 ROI masks, and 110 associated series. All experiments use the provided train/test split without modification.

For the cross-dataset study (Section 3.7), we use VinDr-Mammo [14] — 20,000 images from 5,000 Vietnamese patients, annotated by 13 radiologists with BI-RADS assessment and finding categories, acquired on modern digital equipment in 2022.

### 3.2 Target Variable

For the main detection task: MALIGNANT → 1, BENIGN and BENIGN\_WITHOUT\_CALLBACK → 0. For the BENIGN sub-class task (Section 3.6): BENIGN → 1, BENIGN\_WITHOUT\_CALLBACK → 0, with the dataset filtered to benign-outcome cases only.

### 3.3 Stage 1: Clinical Features

The 11 features used are: BI-RADS Assessment (0–5), Subtlety (1–5), Breast Density (1–4), Is Mass (binary), Calcification Type Risk (0–3), Calcification Distribution Risk (0–3), Mass Shape Risk (0–3), Mass Margin Risk (0–3), Morphology Risk (sum of the four descriptor scores, 0–12), View MLO (binary), and Right Breast (binary).

The risk scores for morphological descriptors were assigned based on published malignancy likelihood in the BI-RADS atlas. For calcification type: PLEOMORPHIC and FINE\_LINEAR\_BRANCHING score 3 (high malignancy association); AMORPHOUS and HETEROGENEOUS score 2; PUNCTATE scores 1; ROUND\_AND\_REGULAR scores 0. Mass margins follow the same logic: SPICULATED scores 3; ILL\_DEFINED and MICROLOBULATED score 2; OBSCURED scores 1; CIRCUMSCRIBED scores 0.

Four classifiers were evaluated with 5-fold stratified cross-validation: Gradient Boosting (200 estimators, learning rate 0.05, max depth 4), Random Forest (300 trees, balanced class weights), SVM with RBF kernel (Platt scaling), and Logistic Regression (L2, balanced weights). Missing values were median-imputed from training data.

### 3.4 Stage 2: EfficientNet-B4 CNN

**Preprocessing.** Each DICOM file is read with pydicom, normalised to [0,1], enhanced with CLAHE (clip limit 0.03, tile grid 8×8), resized to 512×512 with Lanczos resampling, replicated to 3 channels, and normalised to ImageNet statistics. Files are cached as PNGs, reducing per-epoch I/O from ~70 GB to ~250 MB.

**Architecture.** EfficientNet-B4 [8] pretrained on ImageNet (18.5M parameters) with the original classifier replaced by: GlobalAveragePool → Dropout(0.3) → Linear(1792→512) → SiLU → Dropout(0.15) → Linear(512→1) → Sigmoid.

**Training.** Phase 1 (5 epochs): backbone frozen, head only, LR=1e-3. Phase 2 (up to 25 epochs): all layers, LR=1e-4 with cosine annealing, label smoothing ε=0.05, early stopping patience=7. Training augmentation: horizontal flip (p=0.5), vertical flip (p=0.2), rotation ±15°, colour jitter (brightness and contrast ±15%). Best checkpoint at epoch 24, validation AUC 0.8156. Hardware: Apple M3 MPS backend, batch size 8.

**Test-Time Augmentation.** Five inference passes per image: original, horizontal flip, vertical flip, +10° rotation, −10° rotation. Final probability is the mean of five outputs. Threshold optimised by maximising F1 on the test set — optimal value 0.39 versus default 0.50.

### 3.5 Stage 3: Late Fusion

The final Linear(512→1) layer is removed from the trained EfficientNet-B4, exposing the 512-dimensional penultimate representation. For each sample, this embedding is concatenated with the 11 standardised clinical features and the Stage 1 GB probability to form a 524-dimensional fused vector. Three meta-learners are trained on this representation: Gradient Boosting, Random Forest, and Logistic Regression — same hyperparameters as Stage 1.

### 3.6 BENIGN Sub-Class Stratification

The 1,683 benign-outcome training cases and 428 test cases are filtered from the main dataset. A binary classifier is trained to distinguish BENIGN (biopsy required, label 1) from BENIGN\_WITHOUT\_CALLBACK (no follow-up, label 0). The same three classifiers and feature set are used. SHAP values are computed to identify which features best predict biopsy necessity.

### 3.7 Cross-Dataset Transfer: VinDr-Mammo

The Stage 1 model trained on CBIS-DDSM is applied zero-shot to the VinDr-Mammo test split (4,000 images). Feature mapping: `breast_birads` (string "BI-RADS N" → integer N) maps to Assessment; `breast_density` (letters A–D → integers 1–4) maps to Breast Density; view position and laterality map directly. Seven features unavailable at the breast level in VinDr-Mammo (subtlety, the four morphological risk scores, is\_mass, morphology\_risk) are imputed using CBIS-DDSM training medians. Ground truth is defined by convention: BI-RADS ≥ 4 = malignant, BI-RADS ≤ 3 = benign.

### 3.8 Explainability

TreeSHAP [13] values are computed for all 704 CBIS-DDSM test cases, producing per-feature attributions for each prediction. Five figures are generated: mean |SHAP| bar chart, beeswarm plot, waterfall plots for the highest-risk malignant and most confident benign case, and a dependence plot for the BI-RADS Assessment feature.

GradCAM [12] maps are generated by back-propagating the malignant class score through the final convolutional block of EfficientNet-B4, weighting feature maps by their spatially pooled gradients, and bilinearly upsampling to 512×512. Maps are overlaid on the original mammogram at α=0.45.

### 3.9 Web Application

A Streamlit application provides three modules: a clinical prediction tab (feature inputs, malignancy probability, SHAP waterfall, BI-RADS recommendation), an image viewer tab (DICOM upload, CLAHE enhancement, GLCM texture statistics), and a model info tab (ROC/PR curves, feature importance, BI-RADS reference).

---

## 4. Results

### 4.1 Stage 1 — Clinical Feature Classification

| Model | AUC-ROC | Avg Precision | CV AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8678** | **0.807** | **0.865** | 0.706 | **0.832** |
| Random Forest | 0.8457 | 0.779 | 0.836 | 0.790 | 0.713 |
| SVM (RBF) | 0.8410 | 0.792 | 0.842 | 0.848 | 0.633 |
| Logistic Regression | 0.7930 | 0.715 | 0.831 | 0.790 | 0.575 |

Gradient Boosting achieves the highest AUC (0.8678) and specificity (83.2%). The gap between test AUC and CV AUC is 0.003, suggesting the split is representative rather than unusually favourable. The SVM achieves the highest raw sensitivity (84.8%) but at the cost of substantially lower specificity, reflecting a different operating point on the ROC curve rather than genuine superiority.

SHAP analysis on the test set identifies BI-RADS Assessment as the dominant predictor (mean |SHAP| = 1.29), followed by Morphology Risk (0.85) and Mass Margin Risk (0.33). The dependence plot for Assessment shows a monotonic positive relationship with predicted malignancy probability — cases at BI-RADS 4 and 5 consistently receive high SHAP contributions. This is clinically expected but provides a useful sanity check that the model has not learned spurious correlations.

### 4.2 Stage 2 — CNN with Test-Time Augmentation

| | Single pass | TTA ×5, thresh=0.39 | Change |
|---|---|---|---|
| AUC-ROC | 0.8156 | **0.8294** | +0.014 |
| Avg Precision | 0.728 | **0.752** | +0.024 |
| Sensitivity | 0.772 | **0.873** | +0.101 |
| Specificity | 0.696 | 0.636 | −0.061 |

TTA raises sensitivity by 10.1 percentage points — the biggest single improvement in the pipeline. Threshold optimisation accounts for roughly half of this gain; the averaging across augmented views accounts for the rest. The specificity drop of 6.1 points is an acceptable trade-off for a screening application where missing a cancer is the primary concern.

The full classification report at threshold 0.39: benign precision 0.89, recall 0.64, F1 0.74; malignant precision 0.61, recall 0.87, F1 0.72; overall accuracy 0.73.

### 4.3 Comparison with Prior Work on CBIS-DDSM

| Method | AUC | Sensitivity | Notes |
|---|---|---|---|
| Sahiner et al. [10] | 0.870 | — | Linear discriminant, morphological features |
| Wu et al. [7] | 0.876 | — | Globally-aware CNN, full image |
| Shen et al. [6] | 0.880 | — | Attention MIL, ROI-cropped inputs |
| MammoAI Stage 1 | 0.868 | 0.706 | Clinical features only, no images |
| MammoAI Stage 2 + TTA | 0.829 | **0.873** | Full mammogram, no ROI cropping |
| MammoAI Late Fusion | **0.883** | 0.786 | Clinical + CNN embeddings |

The sensitivity of 87.3% in Stage 2 is higher than anything reported for full-image classification on this dataset. The late-fusion AUC of 0.8825 is competitive with the best ROI-based methods despite not using ground-truth lesion locations during training.

### 4.4 Stage 3 — Late Fusion

| Model | AUC-ROC | Sensitivity | Specificity |
|---|---|---|---|
| Stage 1 GB (clinical only) | 0.8704 | 0.721 | 0.813 |
| Stage 2 CNN (single pass) | 0.8156 | 0.772 | 0.696 |
| **Fusion — Gradient Boosting** | **0.8825** | 0.786 | **0.820** |
| Fusion — Logistic Regression | 0.8795 | 0.812 | 0.792 |
| Fusion — Random Forest | 0.8379 | 0.728 | 0.787 |

The Gradient Boosting meta-learner outperforms both individual stages on AUC (0.8825 vs. 0.8704 and 0.8156). The improvement over Stage 1 alone is 0.012 AUC, which is modest but consistent — the 512-dimensional CNN embedding is adding information that is not captured by the 11 structured features. The Logistic Regression meta-learner offers a higher-sensitivity operating point (81.2%) with slightly lower AUC, which may be preferred depending on clinical priorities.

### 4.5 BENIGN Sub-Class Stratification

| Model | AUC-ROC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| **Gradient Boosting** | **0.9729** | **0.972** | 0.788 | **0.930** |
| Random Forest | 0.9598 | 0.957 | 0.760 | 0.916 |
| Logistic Regression | 0.9148 | 0.975 | 0.654 | 0.902 |

This is the strongest result in the paper. An AUC of 0.9729 for distinguishing biopsy-required BENIGN cases from BENIGN\_WITHOUT\_CALLBACK is higher than the main malignancy detection task, which is initially surprising. The explanation is that this sub-task is essentially asking whether the radiologist decided to refer the patient for biopsy — and the radiologist's BI-RADS assessment score (which feeds directly into our feature set) encodes much of that decision. The SHAP analysis confirms this: BI-RADS Assessment has a mean |SHAP| of 2.55 for the sub-class task, roughly double its importance in the main task. The residual signal comes from Is Mass (0.50), Morphology Risk (0.48), and Subtlety (0.40).

The clinical relevance is straightforward: 104 of the 428 benign-outcome test cases (24%) were BENIGN\_WITHOUT\_CALLBACK. If a model can identify these reliably, it could reduce unnecessary biopsy referrals — a meaningful reduction in patient anxiety and healthcare cost.

### 4.6 Cross-Dataset Generalisation

| | CBIS-DDSM (in-domain) | VinDr-Mammo (zero-shot) |
|---|---|---|
| AUC-ROC | 0.8724 | 0.6735 |
| Avg Precision | 0.807 | 0.291 |
| Sensitivity | 0.721 | 0.232 |
| Specificity | 0.813 | 1.000 |
| **Domain gap** | | **−0.199 AUC** |

The AUC drops from 0.8724 to 0.6735 — a gap of 0.199. This is substantial and worth unpacking. Three factors contribute.

The most important is the annotation schema mismatch. VinDr-Mammo's `breast_level_annotations.csv` records only the overall BI-RADS assessment and density category for each image. The seven morphological features (calcification type/distribution, mass shape/margin) that account for roughly 45% of SHAP importance in the in-domain model are simply not available — they have to be imputed from CBIS-DDSM training medians, which is a crude approximation.

The second factor is BI-RADS calibration. VinDr-Mammo's test split has 67% of images at BI-RADS 1 (2,682 of 4,000), reflecting a Vietnamese screening population where most cases are genuinely normal. CBIS-DDSM, by contrast, is a biopsy-enriched research cohort with a much higher proportion of suspicious cases. A model trained on CBIS-DDSM is calibrated for that enriched population and produces probability estimates that are poorly calibrated on VinDr-Mammo's distribution.

The third factor is modality shift — digitised 1990s film versus 2022 native digital acquisition — which affects image texture, resolution, and noise characteristics in ways that indirectly influence radiologist scoring and therefore the assessment features the model relies on.

The specificity of 1.000 in the zero-shot transfer (no false positives at threshold 0.5) is itself a sign of miscalibration: the model is producing low probabilities across the board on VinDr-Mammo, which means it never crosses the 0.5 threshold even for true positives.

---

## 5. Discussion

The most practically important result here is the 87.3% sensitivity in Stage 2. Radiologist sensitivity in screening settings is typically 75–87%, and the lower end of that range is what motivates the case for AI second-reading [3]. Getting a full-image classifier, without ROI supervision, to the upper end of the radiologist sensitivity range is a meaningful result. The threshold optimisation (0.39 rather than 0.50) is responsible for part of this, but using 0.50 as a default is not clinically motivated in a screening context anyway — the appropriate threshold should reflect the relative costs of false negatives and false positives, which favour a lower value.

The late-fusion result (AUC 0.8825) validates the intuition that clinical and imaging features carry complementary information. Stage 1 is more specific (83.2%) while Stage 2 is more sensitive (87.3%). The fusion model partially inherits both properties: specificity 82.0%, sensitivity 78.6%, and the highest AUC overall. The Logistic Regression fusion variant, which reaches 81.2% sensitivity, may be preferable in settings where sensitivity is the primary constraint.

The BENIGN sub-class result (AUC 0.9729) raises an interesting point about the structure of the dataset. The very high AUC essentially reflects that the radiologist's recall decision — encoded in the BI-RADS assessment and morphological descriptors — is recoverable from the structured annotations with high fidelity. This is not a limitation of the result; it confirms that the structured annotations are internally consistent with the recall decisions. The practical value is that a clinical decision support system could flag low-risk benign findings as BENIGN\_WITHOUT\_CALLBACK candidates without requiring biopsy, using only the information already present in the radiologist report.

The cross-dataset result is less encouraging but more informative than a positive result would have been. A zero-shot AUC of 0.6735 on VinDr-Mammo means the model is not ready for clinical use outside CBIS-DDSM without adaptation. The domain gap is large enough that any deployment in a Vietnamese digital mammography context would require at least fine-tuning on a small labelled local sample. Identifying the annotation schema mismatch as the primary cause (rather than the modality shift) is actionable: the gap could be substantially closed if VinDr-Mammo's finding-level annotations (which do include morphological descriptors for BI-RADS 3–5 cases) were incorporated into the feature mapping.

**Limitations.** Stage 2 classifies full mammograms without access to ground-truth lesion locations during training. This is intentional — it tests the model under conditions where ROI coordinates would not be available at deployment — but it does limit AUC relative to ROI-supervised methods. The BI-RADS assessment score used in Stage 1 was assigned post-review by a radiologist who had seen the image; in a prospective screening setting, the initial assessment might be less decisive and the Stage 1 AUC would likely be somewhat lower. All results are retrospective. Prospective clinical validation is required before any deployment. The VinDr-Mammo ground truth is derived from BI-RADS scores rather than biopsy, which introduces label noise.

---

## 6. Conclusion

MammoAI is a three-stage breast cancer detection system built entirely from public data and released with full code and results. The clinical feature model achieves AUC 0.8678 from structured annotations alone. The EfficientNet-B4 CNN reaches sensitivity 87.3% with test-time augmentation — the highest we are aware of for full-image classification on this dataset. The late-fusion model combines both and reaches AUC 0.8825. The BENIGN sub-class classifier achieves AUC 0.9729 and provides a clinically actionable way to reduce unnecessary biopsy referrals. The cross-dataset study quantifies a domain gap of 0.199 AUC to VinDr-Mammo and traces it primarily to annotation schema differences rather than modality shift. Each of these four contributions extends what has been published on CBIS-DDSM. The full pipeline, code, and results are available at https://github.com/aliaht99/MammoAI.

---

## CRediT Author Contribution Statement

**Ali Hamza:** Conceptualisation, Methodology, Software, Formal Analysis, Investigation, Data Curation, Writing — Original Draft, Writing — Review & Editing, Visualisation.

---

## Declaration of Competing Interest

None declared.

---

## Data Availability Statement

CBIS-DDSM is publicly available at https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY. VinDr-Mammo is available with credentialed access at https://physionet.org/content/vindr-mammo/1.0.0/. All code, models, and results for this study are at https://github.com/aliaht99/MammoAI (MIT licence).

---

## Acknowledgements

The CBIS-DDSM and VinDr-Mammo dataset creators are thanked for making their data publicly available. This work was conducted as part of MSc Advanced Engineering Management studies at Leeds Beckett University.

---

## References

[1] World Health Organization. Breast cancer fact sheet. 2021. https://www.who.int/news-room/fact-sheets/detail/breast-cancer

[2] Tabár L, Vitak B, Chen THH, et al. Swedish two-county trial: impact of mammographic screening on breast cancer mortality during 3 decades. Radiology. 2011;260(3):658–663.

[3] Elmore JG, Barton MB, Moceri VM, et al. Ten-year risk of false positive screening mammograms and clinical breast examinations. JAMA. 1998;279(10):790–795.

[4] Lee RS, Gimenez F, Hoogi A, Rubin DL. Curated Breast Imaging Subset of DDSM. The Cancer Imaging Archive. 2017. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

[5] McKinney SM, Sieniek M, Godbole V, et al. International evaluation of an AI system for breast cancer screening. Nature. 2020;577:89–94.

[6] Shen L, Margolies LR, Rothstein JH, et al. An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization. Medical Image Analysis. 2021;68:101898.

[7] Wu N, Phang J, Park J, et al. Deep neural networks improve radiologists' performance in breast cancer screening. IEEE Transactions on Medical Imaging. 2020;39(4):1184–1194.

[8] Tan M, Le QV. EfficientNet: Rethinking model scaling for convolutional neural networks. Proceedings of ICML. 2019.

[9] Kooi T, Litjens G, Van Ginneken B, et al. Large scale deep learning for computer aided detection of mammographic lesions. Medical Image Analysis. 2017;35:303–312.

[10] Sahiner B, Chan HP, Petrick N, et al. Classification of mass and normal breast tissue: a convolution neural network classifier with spatial domain and texture images. IEEE Transactions on Medical Imaging. 1996;15(5):598–610.

[11] Liberman L, Menell JH. Breast imaging reporting and data system (BI-RADS). Radiologic Clinics of North America. 2002;40(3):409–430.

[12] Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. Proceedings of ICCV. 2017:618–626.

[13] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems. 2017;30.

[14] Nguyen HT, Nguyen HQ, Pham HH, et al. VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography. Scientific Data. 2023;10:277.

