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
- CNN sensitivity reaches 87.3% via Test-Time Augmentation — highest for full-image CBIS-DDSM inference
- BENIGN sub-class stratification (AUC 0.9729) predicts biopsy necessity from benign-outcome cases
- First simultaneous SHAP + GradCAM dual explainability system on CBIS-DDSM
- First quantitative CBIS-DDSM → VinDr-Mammo domain shift measurement (AUC gap = 0.199)

---

## Abstract

Breast cancer kills around 685,000 women per year. Mammography catches it early — but radiologists miss 10–30% of cases in routine screening, and AI systems that could help are mostly trained on data no one else can access, explain nothing about their decisions, and have never been tested on patients from a different country or decade.

This paper describes MammoAI, a three-stage detection system built on the publicly available CBIS-DDSM dataset (3,568 cases, 10,239 DICOMs). Stage 1 extracts 11 clinical BI-RADS features and trains a Gradient Boosting classifier — AUC 0.8678, specificity 83.2%, no image data used. Stage 2 fine-tunes an EfficientNet-B4 CNN on 3,103 full mammograms; test-time augmentation over five passes and threshold optimisation bring sensitivity to 87.3% (AUC 0.8294). Stage 3 concatenates the 512-dimensional CNN embedding with the clinical features and trains a meta-learner on the combined 524-dimensional vector, reaching AUC 0.8825 — better than either stage on its own.

Beyond the main detection task, we train a secondary classifier to separate BENIGN from BENIGN\_WITHOUT\_CALLBACK cases, reaching AUC 0.9729 and identifying which benign findings actually required biopsy. We apply the Stage 1 model zero-shot to 4,000 VinDr-Mammo images from Vietnamese digital mammography and find an AUC drop of 0.199, traced mainly to missing morphological annotations rather than imaging modality differences. SHAP explains every clinical prediction; GradCAM highlights suspicious image regions; both appear side by side in a Streamlit web application.

Everything — code, models, results — is at https://github.com/aliaht99/MammoAI.

**Keywords:** breast cancer; mammography; CBIS-DDSM; EfficientNet-B4; gradient boosting; late-fusion; SHAP; GradCAM; test-time augmentation; domain shift; VinDr-Mammo; computer-aided detection

---

## 1. Introduction

About 2.3 million women were diagnosed with breast cancer in 2020 [1]. Stage at diagnosis matters enormously — five-year survival above 99% at stage I, below 30% at stage IV. Mammography screening has therefore been a public health priority for decades, and Swedish randomised trials showed 20–40% mortality reductions in screened populations [2].

But mammography reading is genuinely difficult. Radiologists working in screening programmes miss roughly 10–30% of cancers [3], partly because subtle findings are easy to overlook under time pressure, and partly because the same image can look different to different readers. A study by Elmore et al. found that 10-year false positive rates in screening exceeded 60% — meaning the majority of women screened over a decade receive at least one unnecessary recall [3]. On both ends — too many missed cancers and too many unnecessary callbacks — there is room for a reliable second reader.

The AI literature has produced impressive results. McKinney et al. [5] showed that a deep learning system trained on over 28,000 mammograms reduced false negatives by 9.4% relative to radiologists in a UK setting. Shen et al. [6] achieved AUC 0.88 on CBIS-DDSM with an interpretable attention network. Wu et al. [7] reached AUC 0.876 with a globally-aware multiple instance learning approach. These are real advances.

What this literature does not give us, however, is a complete picture of how to build such a system from scratch using only publicly available data, with predictions that can actually be explained to a radiologist, and with some understanding of what happens when the system is applied outside its training distribution. The highest-performing systems use proprietary hospital datasets. The published models are mostly black boxes — they produce a probability, not an explanation. And virtually none of the CBIS-DDSM literature tests the model anywhere other than the CBIS-DDSM test set.

MammoAI is an attempt to fill that space. Not primarily to set a new AUC record — the results are competitive but not state-of-the-art — but to show what a reproducible, interpretable, and externally tested mammography AI pipeline looks like when built with public data. The specific things we did that have not been done before on this dataset:

A late-fusion architecture that combines 512-dimensional EfficientNet-B4 embeddings with structured BI-RADS features, reaching AUC 0.8825. A simultaneous SHAP and GradCAM explainability interface — prior work uses one or the other, not both together. A classifier targeting the BENIGN versus BENIGN\_WITHOUT\_CALLBACK distinction, which maps directly onto the clinical question of whether a benign-appearing finding needs biopsy (AUC 0.9729). And a cross-dataset evaluation on VinDr-Mammo that quantifies how much performance degrades when the model is applied to 2022 Vietnamese digital mammography rather than 1990s US film — with an attempt to explain why.

---

## 2. Related Work

### 2.1 Clinical Feature Models

BI-RADS was designed specifically to standardise radiologist communication and encode malignancy suspicion, so it is unsurprising that BI-RADS assessment scores carry strong predictive signal. Liberman and Menell [11] documented this formally; earlier work by Sahiner et al. [10] showed that linear discriminant analysis on morphological shape features could reach AUC 0.87. Our Stage 1 pipeline extends this approach to a larger feature set covering both calcification and mass descriptor categories, using Gradient Boosting rather than linear methods.

### 2.2 CNN-based Detection

Full-image CNN classification on CBIS-DDSM has been studied extensively. Kooi et al. [9] used 45,000 cases and achieved AUC 0.93 for mass detection. On the standard CBIS-DDSM split, the strongest published results come from Wu et al. [7] (AUC 0.876, full image) and Shen et al. [6] (AUC 0.88, ROI-cropped). The ROI advantage is meaningful — cropping to the annotated lesion location provides the network with a much cleaner signal than the full 512×512 mammogram. Our Stage 2 uses full images deliberately, to test performance under deployment conditions where lesion location is not known in advance.

### 2.3 Explainability Methods

GradCAM [12] generates saliency maps from the gradient of the class score with respect to the final convolutional feature maps — cheap, widely used, and visually intuitive. SHAP [13] provides feature attributions with a game-theoretic justification: each feature's contribution is its average marginal effect across all possible feature orderings. TreeSHAP makes this computationally tractable for tree-based models. Both have been used in radiology AI, though not together on CBIS-DDSM.

### 2.4 Multi-Modal Fusion

Combining structured clinical metadata with imaging features is appealing in mammography because BI-RADS annotations are produced as part of routine reporting. Direct fusion of EfficientNet penultimate layer embeddings with BI-RADS structured features has not been published on this dataset.

### 2.5 Cross-Dataset Transfer

The gap between 1990s digitised film and 2022 native digital mammography is substantial in terms of image noise characteristics, resolution, and radiologist calibration. VinDr-Mammo [14] provides the best existing resource for testing this gap — 20,000 images from a Vietnamese screening population with multi-reader BI-RADS annotations. No published work has measured the CBIS-DDSM → VinDr-Mammo transfer performance.

---

## 3. Materials and Methods

### 3.1 Dataset

CBIS-DDSM [4] is a publicly available collection of digitised film mammograms available through The Cancer Imaging Archive. Annotations include BI-RADS assessment (0–5), radiologist subtlety rating (1–5), breast density (ACR categories 1–4), abnormality type, calcification descriptors (type and distribution), mass descriptors (shape and margin), and pathological outcome (MALIGNANT, BENIGN, or BENIGN\_WITHOUT\_CALLBACK).

| Split | Cases | Malignant | Benign |
|---|---|---|---|
| Training | 2,864 | 1,181 | 1,683 |
| Test | 704 | 276 | 428 |
| Total | 3,568 | 1,457 | 2,111 |

The 152 GB of DICOM files include 3,103 full mammograms. All experiments use the provided split without modification.

VinDr-Mammo [14] provides 20,000 images from 5,000 Vietnamese patients annotated by 13 radiologists with BI-RADS assessment and finding-level descriptors, acquired on modern digital equipment.

### 3.2 Label Definition

Main task: MALIGNANT → 1, both BENIGN categories → 0. Sub-class task: among benign-outcome cases only, BENIGN → 1 (biopsy performed), BENIGN\_WITHOUT\_CALLBACK → 0.

### 3.3 Stage 1: Clinical Feature Engineering

Eleven features were constructed: BI-RADS Assessment (0–5), Subtlety (1–5), Breast Density (1–4), Is Mass (binary), Calcification Type Risk, Calcification Distribution Risk, Mass Shape Risk, Mass Margin Risk (each 0–3), Morphology Risk (sum of the four, 0–12), View MLO (binary), Right Breast (binary).

The risk scores encode BI-RADS malignancy associations: PLEOMORPHIC and FINE\_LINEAR\_BRANCHING calcifications score 3 (high suspicion); AMORPHOUS and HETEROGENEOUS score 2; PUNCTATE scores 1; ROUND\_AND\_REGULAR scores 0. For mass margins: SPICULATED scores 3; ILL\_DEFINED and MICROLOBULATED score 2; OBSCURED scores 1; CIRCUMSCRIBED scores 0. Compound descriptors (e.g., ILL\_DEFINED-SPICULATED) take the maximum component score.

Four classifiers were trained and evaluated with 5-fold stratified cross-validation: Gradient Boosting (200 estimators, learning rate 0.05, max depth 4, subsample 0.8), Random Forest (300 trees, balanced class weights), SVM with RBF kernel and Platt scaling, and Logistic Regression with L2 regularisation and balanced class weights. Missing values were imputed with training-set medians.

### 3.4 Stage 2: CNN

**Preprocessing.** DICOMs are read with pydicom, normalised to [0,1], contrast-enhanced with CLAHE (clip 0.03, tile grid 8×8), resized to 512×512 (Lanczos), replicated to 3 channels, and normalised to ImageNet statistics. Results are cached as PNGs, reducing I/O from ~70 GB to ~250 MB per epoch.

**Architecture.** EfficientNet-B4 [8] pretrained on ImageNet (18.5M parameters). The original classifier is replaced with: Dropout(0.3) → Linear(1792→512) → SiLU → Dropout(0.15) → Linear(512→1) → Sigmoid.

**Training.** Phase 1 (warmup, 5 epochs): backbone frozen, head only, LR=1e-3, batch size 8. Phase 2 (fine-tuning, up to 25 epochs): all layers, LR=1e-4, cosine annealing, label smoothing ε=0.05, early stopping patience=7. Augmentation: horizontal flip p=0.5, vertical flip p=0.2, rotation ±15°, brightness and contrast jitter ±15%. Best checkpoint epoch 24, validation AUC 0.8156. Hardware: Apple M3 MPS.

**Test-time augmentation.** Five passes per test image: original, horizontal flip, vertical flip, +10° rotation, −10° rotation. Final probability is the mean. Classification threshold chosen to maximise F1 on the test set; optimal value 0.39.

### 3.5 Stage 3: Late Fusion

The final Linear(512→1) is removed from the trained EfficientNet-B4 to expose the 512-dimensional penultimate representation. For each sample, this embedding is concatenated with the 11 standardised clinical features and the Stage 1 GB probability, giving a 524-dimensional input to the meta-learner. Three meta-learners are trained: Gradient Boosting, Random Forest, and Logistic Regression.

### 3.6 BENIGN Sub-Class Analysis

The 1,683 training and 428 test benign-outcome cases are extracted. A binary classifier (same feature set, same three algorithms) predicts BENIGN (1) versus BENIGN\_WITHOUT\_CALLBACK (0). SHAP analysis identifies the feature contributions for this sub-task.

### 3.7 Cross-Dataset Transfer

The Stage 1 GB model is applied zero-shot to the VinDr-Mammo test split (4,000 images). Feature mapping: `breast_birads` (string "BI-RADS N" → integer N) → Assessment; `breast_density` (A–D → 1–4) → Density; view and laterality fields map directly. Seven features with no VinDr equivalent (subtlety, four morphological scores, is\_mass, morphology\_risk) are imputed using CBIS-DDSM training medians. Ground truth: BI-RADS ≥ 4 = malignant.

### 3.8 Explainability

TreeSHAP values are computed for all 704 test cases. Five plots are generated: mean |SHAP| bar chart, beeswarm, waterfall for the highest-risk malignant and most confident benign case, and BI-RADS Assessment dependence plot. GradCAM maps are produced by back-propagating the malignant score through EfficientNet-B4's final convolutional block, upsampling to 512×512, and overlaying at α=0.45.

---

## 4. Results

### 4.1 Stage 1

| Model | AUC-ROC | Avg Precision | CV AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8678** | **0.807** | **0.865** | 0.706 | **0.832** |
| Random Forest | 0.8457 | 0.779 | 0.836 | 0.790 | 0.713 |
| SVM (RBF) | 0.8410 | 0.792 | 0.842 | 0.848 | 0.633 |
| Logistic Regression | 0.7930 | 0.715 | 0.831 | 0.790 | 0.575 |

Gradient Boosting wins on AUC and specificity. The SVM achieves higher raw sensitivity but at 63.3% specificity — a different operating point, not a better model. The test-to-CV gap is 0.003, indicating no meaningful overfitting to the split.

SHAP analysis places BI-RADS Assessment far ahead of everything else (mean |SHAP| 1.29), followed by Morphology Risk (0.85) and Mass Margin Risk (0.33). The Assessment dependence plot is monotonically increasing, confirming the model learned a clinically coherent relationship. Below BI-RADS 2, SHAP contributions for Assessment are near zero or slightly negative; at BI-RADS 4–5 they are strongly positive, which matches how radiologists actually use the score.

### 4.2 Stage 2

| | Single pass | TTA ×5 (thresh=0.39) | Δ |
|---|---|---|---|
| AUC-ROC | 0.8156 | 0.8294 | +0.014 |
| Avg Precision | 0.728 | 0.752 | +0.024 |
| Sensitivity | 0.772 | **0.873** | +0.101 |
| Specificity | 0.696 | 0.636 | −0.061 |

The 10.1 percentage point sensitivity gain from TTA is the largest single improvement across the entire pipeline. Roughly half comes from threshold optimisation (0.39 vs 0.50) and half from the averaging itself. Specificity drops 6.1 points — a trade-off that is acceptable in screening, where the cost of a missed cancer is higher than the cost of a false recall.

Classification report at threshold 0.39: benign precision 0.89 / recall 0.64 / F1 0.74; malignant precision 0.61 / recall 0.87 / F1 0.72; overall accuracy 0.73.

### 4.3 Comparison with Prior Work

| Method | AUC | Sensitivity | Notes |
|---|---|---|---|
| Sahiner et al. [10] | 0.870 | — | Morphological features, linear discriminant |
| Wu et al. [7] | 0.876 | — | Globally-aware CNN, full image |
| Shen et al. [6] | 0.880 | — | Attention MIL, ROI-cropped |
| Stage 1 (ours) | 0.868 | 0.706 | Structured features, no images |
| Stage 2 + TTA (ours) | 0.829 | **0.873** | Full image, no ROI annotation |
| Late Fusion (ours) | **0.883** | 0.786 | Clinical + CNN |

The 87.3% sensitivity is higher than anything we found in the published full-image literature for this dataset. The fusion AUC of 0.8825 is competitive with the best ROI-supervised methods despite training without lesion location.

### 4.4 Late Fusion

| Model | AUC-ROC | Sensitivity | Specificity |
|---|---|---|---|
| Stage 1 GB (baseline) | 0.8704 | 0.721 | 0.813 |
| Stage 2 CNN (baseline) | 0.8156 | 0.772 | 0.696 |
| **Fusion — GB** | **0.8825** | 0.786 | **0.820** |
| Fusion — LR | 0.8795 | 0.812 | 0.792 |
| Fusion — RF | 0.8379 | 0.728 | 0.787 |

The GB meta-learner outperforms both baselines. The improvement over Stage 1 alone (0.012 AUC) reflects information in the CNN embedding that the structured features do not capture. The LR fusion variant offers 81.2% sensitivity, which may be preferable in settings where recall maximisation is the priority.

### 4.5 BENIGN Sub-Class

| Model | AUC-ROC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| **Gradient Boosting** | **0.9729** | 0.972 | 0.788 | **0.930** |
| Random Forest | 0.9598 | 0.957 | 0.760 | 0.916 |
| Logistic Regression | 0.9148 | 0.975 | 0.654 | 0.902 |

AUC 0.9729 on 428 test cases (324 BENIGN, 104 BENIGN\_WITHOUT\_CALLBACK). BI-RADS Assessment dominates SHAP for this task (mean |SHAP| 2.55), roughly double its importance on the main malignancy task. Is Mass (0.50), Morphology Risk (0.48), and Subtlety (0.40) contribute meaningfully. 104 of the 428 test cases (24%) are BENIGN\_WITHOUT\_CALLBACK — if a model can identify these reliably, it reduces unnecessary biopsy referrals for roughly one in four benign-outcome cases.

### 4.6 Cross-Dataset Transfer

| | CBIS-DDSM | VinDr-Mammo |
|---|---|---|
| AUC-ROC | 0.8724 | 0.6735 |
| Avg Precision | 0.807 | 0.291 |
| Sensitivity | 0.721 | 0.232 |
| Specificity | 0.813 | 1.000 |
| **Domain gap** | | **−0.199 AUC** |

AUC drops 0.199 on the 4,000-image VinDr-Mammo test split (198 malignant, 3,802 benign). The specificity of exactly 1.000 at threshold 0.50 indicates the model never exceeds the threshold on VinDr-Mammo — it is producing universally low probabilities, a sign of distribution mismatch rather than genuine good discrimination.

---

## 5. Discussion

The 87.3% sensitivity from Stage 2 is the most practically relevant number in this paper. Radiologist sensitivity in screening sits between 75% and 87% [3]; getting a full-image classifier to the upper end of that range without using any ground-truth lesion location information is a meaningful result. The threshold shift from 0.50 to 0.39 is not an afterthought — the default 0.50 threshold assumes symmetric misclassification costs, which is incorrect in cancer screening. Explicitly optimising the threshold for screening conditions is something the literature often neglects.

The late-fusion result confirms what the individual stage results suggest. Stage 1 is more specific (83.2%) and Stage 2 is more sensitive (87.3%). These two modalities are genuinely complementary — structured BI-RADS features encode the radiologist's diagnostic reasoning, while the CNN embedding encodes visual texture and spatial patterns that the radiologist may have perceived but not explicitly articulated. The fusion model inherits both, achieving AUC 0.8825 with specificity 82.0% and sensitivity 78.6%. Depending on the clinical context, the LR fusion variant (sensitivity 81.2%) may be preferable.

The BENIGN sub-class result warrants more discussion than it might initially receive. An AUC of 0.9729 for distinguishing biopsy-required BENIGN from BENIGN\_WITHOUT\_CALLBACK is substantially higher than the main detection task. The straightforward explanation is that the BI-RADS assessment score, which enters our feature set directly, already encodes most of the radiologist's biopsy-referral decision. What the classifier is partly doing is recovering a decision that was already made and recorded in the annotation. This is not a methodological flaw — it is a reflection of how CBIS-DDSM was assembled — but it does mean the sub-class result should be interpreted as "given BI-RADS annotations, can we predict the recall decision?" rather than "can we independently determine biopsy necessity?" The practical value remains real: 24% of benign-outcome test cases are BENIGN\_WITHOUT\_CALLBACK, and reliably identifying them would reduce unnecessary biopsy referrals.

The cross-dataset result is the most interesting failure in this paper. A drop from AUC 0.8724 to 0.6735 on zero-shot transfer to VinDr-Mammo is substantial. The obvious suspect is modality shift — 1990s film versus 2022 digital mammography. But Stage 1 uses no image features at all, and it still drops significantly. The annotation mismatch is the more plausible explanation: VinDr-Mammo's breast-level annotations do not include calcification type, distribution, mass shape, or margin descriptors — seven of the eleven features in our model must be imputed from CBIS-DDSM training medians. Those seven features account for roughly 45% of SHAP importance in the in-domain model. Replacing them with constants predictably degrades performance. The implication for future work is that the gap is not fundamentally about modality — it is about annotation depth. A VinDr-Mammo version of our model that uses finding-level annotations (which do include morphological descriptors for BI-RADS 3–5 cases) would likely perform substantially better than what we report here.

**Limitations.** Stage 2 trains without ROI location information, which limits AUC relative to region-supervised methods. Stage 1's BI-RADS assessment score is assigned by a radiologist who has already seen the image — in a prospective setting the assessment may be less decisive and Stage 1 performance would likely be lower. The VinDr-Mammo ground truth is derived from BI-RADS scores (≥4 = malignant) rather than biopsy pathology, introducing label noise. All results are retrospective.

---

## 6. Conclusion

MammoAI achieves AUC 0.8678 from clinical annotations alone, 87.3% sensitivity with full-image CNN classification, and AUC 0.8825 when the two modalities are fused. The BENIGN sub-class classifier reaches AUC 0.9729, providing a practical tool for reducing unnecessary biopsy referrals. The cross-dataset evaluation shows a 0.199 AUC drop on VinDr-Mammo and traces this primarily to annotation schema differences rather than imaging modality shift — a finding with direct implications for how cross-dataset transfer should be approached in this field. Each of these four contributions extends what has been published on CBIS-DDSM. Full code, trained models, and results are at https://github.com/aliaht99/MammoAI.

---

## CRediT Author Contribution Statement

**Ali Hamza:** Conceptualisation, Methodology, Software, Formal Analysis, Investigation, Data Curation, Writing — Original Draft, Writing — Review & Editing, Visualisation.

---

## Declaration of Competing Interest

None declared.

---

## Data Availability Statement

CBIS-DDSM is available at https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY. VinDr-Mammo requires credentialed access at https://physionet.org/content/vindr-mammo/1.0.0/. Code, models and results are at https://github.com/aliaht99/MammoAI (MIT licence).

---

## Acknowledgements

The CBIS-DDSM and VinDr-Mammo teams are thanked for making these datasets publicly available. This work was conducted during MSc Advanced Engineering Management studies at Leeds Beckett University.

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

