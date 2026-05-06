# MammoAI — Breast Cancer Detection from Mammograms

> **Early-stage breast cancer detection using clinical features and deep learning on the CBIS-DDSM dataset.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CBIS--DDSM-orange.svg)](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)

---

## Overview

MammoAI is a two-stage breast cancer detection system trained on the **CBIS-DDSM** (Curated Breast Imaging Subset of Digital Database for Screening Mammography):

| Stage | Approach | AUC-ROC |
|---|---|---|
| **Stage 1** (current) | Gradient Boosting on clinical/radiological features | **0.868** |
| **Stage 2** (in progress) | EfficientNet-B4 CNN on raw DICOM mammogram images | — |

---

## Dataset

**CBIS-DDSM** — publicly available via [The Cancer Imaging Archive (TCIA)](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)

| Split | Calcification | Mass | Total |
|---|---|---|---|
| Train | 1,546 | 1,318 | **2,864** |
| Test  | 326   | 378  | **704**   |

**Classes:** `MALIGNANT` (1) vs `BENIGN / BENIGN_WITHOUT_CALLBACK` (0)

---

## Project Structure

```
MammoAI/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md
├── LICENSE
│
├── src/
│   ├── cancer_detection.py   # Stage 1: clinical feature ML pipeline
│   ├── save_model.py         # Train & serialize Gradient Boosting model
│   └── cnn_pipeline.py       # Stage 2: CNN deep learning pipeline (coming)
│
├── models/
│   └── gb_model.pkl          # Trained Gradient Boosting model
│
├── data/
│   └── csv/                  # CBIS-DDSM CSV annotations (lightweight)
│       ├── calc_case_description_train_set.csv
│       ├── calc_case_description_test_set.csv
│       ├── mass_case_description_train_set.csv
│       └── mass_case_description_test_set.csv
│       NOTE: Raw DICOM images (~152 GB) are NOT committed — download from TCIA
│
├── results/                  # Generated plots and metrics
│   ├── roc_pr_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── model_summary.csv
│
├── paper/
│   ├── mammoai_paper.md      # Research paper draft (Markdown → LaTeX)
│   └── figures/              # High-res figures for publication
│
└── notebooks/                # Jupyter notebooks for exploration
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/MammoAI.git
cd MammoAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download CBIS-DDSM from TCIA (free account required)
#    https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
#    Place DICOM folders under: manifest-.../CBIS-DDSM/

# 4. Train the clinical features model
python src/save_model.py

# 5. Launch the web app
streamlit run app.py
```

---

## Results (Stage 1 — Clinical Features)

| Model | AUC-ROC | Sensitivity | Specificity | Avg Precision |
|---|---|---|---|---|
| **Gradient Boosting** | **0.868** | 0.706 | 0.832 | 0.807 |
| Random Forest | 0.846 | 0.790 | 0.713 | 0.779 |
| SVM (RBF) | 0.841 | 0.848 | 0.633 | 0.792 |
| Logistic Regression | 0.793 | 0.790 | 0.575 | 0.715 |

---

## How to Reproduce

```bash
# Stage 1 — feature engineering + ML
python src/cancer_detection.py

# Web app
streamlit run app.py
```

---

## Hardware

Developed and tested on **Apple MacBook Air M3** (MPS backend for PyTorch).  
Stage 2 CNN training uses `torch.device("mps")` automatically.

---

## Citation

If you use this work, please cite:

```bibtex
@software{mammoai2026,
  author    = {Hamza, Ali},
  title     = {MammoAI: Breast Cancer Detection from Mammograms using CBIS-DDSM},
  year      = {2026},
  url       = {https://github.com/YOUR_USERNAME/MammoAI}
}
```

**Dataset citation:**
```bibtex
@article{cbisddsm2017,
  author  = {Lee, Rebecca Sawyer and Gimenez, Francisco and Hoogi, Assaf and Rubin, Daniel},
  title   = {Curated Breast Imaging Subset of DDSM (CBIS-DDSM)},
  journal = {The Cancer Imaging Archive},
  year    = {2017},
  doi     = {10.7937/K9/TCIA.2016.7O02S9CY}
}
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Acknowledgements

- Dataset: [TCIA CBIS-DDSM](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)
- Original DDSM: University of South Florida
