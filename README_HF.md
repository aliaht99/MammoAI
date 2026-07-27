---
title: MammoAI — Breast Cancer Detection
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: mammo_doctor.py
pinned: false
license: mit
short_description: Interpretable multi-modal breast cancer detection (research demo)
tags:
  - medical-imaging
  - mammography
  - breast-cancer
  - explainable-ai
  - healthcare
---

# 🏥 MammoAI — Multi-Modal Breast Cancer Detection

> ⚕️ **Research and education only.** This is **not a medical device**, is not
> cleared by any regulator, and must **never** be used for clinical diagnosis or
> patient care. Its outputs cannot replace a qualified radiologist.
> **Do not upload identifiable patient data.**

An interpretable AI system that estimates breast-cancer risk either from a
radiologist's **BI-RADS findings** or from a **mammogram image**, and explains
*why* it reached that conclusion.

## How to use it

**Option 1 — From clinical findings (no image needed, fastest)**
1. Open the **Clinical Form** tab.
2. Choose the abnormality (Mass or Calcification) and fill in the BI-RADS
   details — assessment, subtlety, shape, margins, density.
3. Press **Run Full Analysis**.
4. You get the malignancy probability, risk band, a biopsy recommendation, and a
   **SHAP chart** showing which features drove the prediction.

**Option 2 — From a mammogram image**
1. In the left sidebar, upload a mammogram (PNG, JPG or DICOM).
2. The **AI Analysis** tab runs automatically and shows the probability gauge and
   uncertainty estimates.

No sample image? The GitHub repo ships 7 demo mammograms in `demo_images/`,
each with the expected result documented in `DEMO_CHEATSHEET.txt`.

## Tabs

| Tab | What it does |
|---|---|
| **AI Analysis** | Image-based prediction, uncertainty, GradCAM heat-map |
| **Image Viewer** | Enhance and inspect the mammogram (CLAHE, histogram, texture) |
| **Clinical Form** | Findings-based prediction + SHAP explanation |
| **Report** | Printable summary of the analysis |
| **Datasets & Models** | The models, their scores, and calibration plots |

## What's running here

| Stage | Method | AUC-ROC |
|---|---|---|
| Stage 1 | Gradient Boosting — 11 clinical BI-RADS features | 0.8678 |
| Stage 1+ | Multi-dataset GB (CBIS-DDSM + VinDr-Mammo, 18,864 cases) | 0.9925 |
| Sub-class | BENIGN biopsy-need stratification | 0.9729 |
| Ensemble | Calibrated ensemble (ECE 0.052) | 0.8722 |

> **Note on Stage 2:** the EfficientNet-B4 image CNN (AUC 0.8294, 87.3% sensitivity)
> is part of the full pipeline but its 74 MB checkpoint is not bundled in this
> Space, so the CNN and GradCAM panels show as unavailable here. Everything else —
> clinical prediction, sub-class stratification, the calibrated ensemble and SHAP
> explanations — is fully live. To run the CNN, clone the GitHub repo and train it.

## Data

Trained on [CBIS-DDSM](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY) (TCIA) and
VinDr-Mammo. No patient data is stored by this app; uploads are processed in
memory for the duration of the session only.

## Source code

**https://github.com/aliaht99/MammoAI** — MIT licensed.

## Citation

```bibtex
@article{hamza2026mammoai,
  author  = {Hamza, Ali},
  title   = {MammoAI: A Multi-Modal Interpretable Framework for Breast Cancer
             Detection — Fusing Clinical Features with Deep Learning on CBIS-DDSM},
  year    = {2026},
  note    = {Under review},
  url     = {https://github.com/aliaht99/MammoAI}
}
```
