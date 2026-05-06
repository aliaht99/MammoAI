# MammoAI — Multi-Modal Breast Cancer Detection

> **A three-stage interpretable AI system for breast cancer detection, fusing clinical BI-RADS features with deep learning on CBIS-DDSM mammography.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CBIS--DDSM-purple.svg)](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)

---

## Results

| Stage | Method | AUC-ROC | Sensitivity | Specificity |
|---|---|---|---|---|
| **Stage 1** | Gradient Boosting — 11 clinical features | **0.8678** | 72.1% | **83.2%** |
| **Stage 2** | EfficientNet-B4 + TTA ×5 | **0.8294** | **87.3%** | 63.6% |
| **Stage 3** | Late-Fusion (512-dim CNN + clinical) | **0.8825** | 78.6% | 82.0% |
| **Sub-class** | BENIGN risk stratification | **0.9729** | 97.2% | 78.9% |

---

## Four Novel Contributions

1. **Dual Explainability** — SHAP (clinical stage) + GradCAM (image stage) shown simultaneously
2. **Late-Fusion Architecture** — 512-dim CNN embeddings + 11 clinical features → AUC 0.8825
3. **BENIGN Sub-Class Stratification** — predicts biopsy need from benign cases (AUC 0.9729)
4. **Cross-Dataset Generalisation** — zero-shot transfer protocol to VinDr-Mammo

---

## Architecture

```
CBIS-DDSM CSV ──► Stage 1: Gradient Boosting ──► AUC 0.8678 + SHAP explanations
                      │
CBIS-DDSM DICOM ──► Stage 2: EfficientNet-B4 ──► AUC 0.8294 + GradCAM heatmaps
                      │
                 Stage 3: Late Fusion ──────────► AUC 0.8825 (best overall)
                   [11 clinical + 512 CNN + 1 GB_prob = 524-dim → GB meta-learner]
```

---

## Project Structure

```
MammoAI/
├── app.py                        Streamlit web application
├── requirements.txt
│
├── src/
│   ├── cancer_detection.py       Stage 1: full ML pipeline + plots
│   ├── save_model.py             Train & serialize GB model
│   ├── shap_analysis.py          SHAP explainability (Contribution 1)
│   ├── benign_subclass.py        BENIGN sub-class analysis (Contribution 3)
│   └── cross_dataset.py          VinDr-Mammo cross-dataset study (Contribution 4)
│
├── stage2_cnn/
│   ├── config.py                 All hyperparameters
│   ├── dataset.py                DICOM loader + CLAHE preprocessing
│   ├── model.py                  EfficientNet-B4 architecture
│   ├── train.py                  Two-phase training + resume
│   ├── evaluate.py               TTA evaluation + GradCAM
│   ├── predict.py                Single-image inference
│   └── fusion.py                 Late-fusion pipeline (Contribution 2)
│
├── models/
│   └── gb_model.pkl              Trained Stage 1 model
│
├── results/
│   ├── shap/                     SHAP plots (5 figures)
│   ├── benign_subclass/          Sub-class analysis plots
│   ├── cross_dataset/            Domain shift analysis
│   └── stage2_cnn/results/       CNN evaluation + GradCAM + fusion plots
│
├── paper/
│   └── mammoai_paper.md          Full research paper (14 references)
│
└── data/csv/                     CBIS-DDSM CSV annotations (lightweight)
    NOTE: Raw DICOM images (~152 GB) are NOT committed — download from TCIA
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/aliaht99/MammoAI.git
cd MammoAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download CBIS-DDSM from TCIA (free account required)
#    https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

# 4. Run Stage 1 (no images needed — CSV only)
python src/cancer_detection.py

# 5. Run SHAP explainability
python src/shap_analysis.py

# 6. Run BENIGN sub-class analysis
python src/benign_subclass.py

# 7. Train Stage 2 CNN (requires DICOM images + GPU/MPS)
cd stage2_cnn && python train.py

# 8. Evaluate Stage 2 with TTA
python evaluate.py

# 9. Run late-fusion (Stage 3)
python fusion.py

# 10. Launch web app
cd .. && streamlit run app.py
```

---

## Stage 2 CNN Training Details

| Hyperparameter | Value |
|---|---|
| Backbone | EfficientNet-B4 (ImageNet pretrained) |
| Image size | 512 × 512 |
| Batch size | 8 |
| Phase 1 — Warmup | 5 epochs, LR=1e-3, backbone frozen |
| Phase 2 — Fine-tune | up to 25 epochs, LR=1e-4, cosine annealing |
| Early stopping patience | 7 epochs |
| Label smoothing | ε = 0.05 |
| Best checkpoint | Epoch 24, val AUC = 0.8156 |
| TTA passes | 5 (original + H-flip + V-flip + ±10° rotation) |
| Optimal threshold | 0.39 |
| Hardware | Apple M3 MPS |

**Resume interrupted training:**
```bash
cd stage2_cnn && python train.py --resume
```

---

## SHAP Feature Importance (Stage 1)

| Feature | Mean |SHAP| |
|---|---|
| BI-RADS Assessment | 1.29 |
| Morphology Risk | 0.85 |
| Mass Margin Risk | 0.33 |
| Subtlety | 0.26 |
| Calc Type Risk | 0.22 |

---

## Cross-Dataset Protocol (VinDr-Mammo)

```bash
# 1. Get free credentialed access at physionet.org
# 2. Complete CITI training + sign data use agreement
# 3. Download CSVs only (~2 MB):
#    curl -u YOUR_USERNAME https://physionet.org/files/vindr-mammo/1.0.0/breast-level_annotations.csv \
#         -o vindr-mammo/breast-level_annotations.csv
# 4. Run cross-dataset study:
python src/cross_dataset.py --vindr /path/to/vindr-mammo/1.0.0
```

---

## Citation

```bibtex
@article{hamza2026mammoai,
  author  = {Hamza, Ali},
  title   = {MammoAI: A Multi-Modal Interpretable Framework for Breast Cancer
             Detection — Fusing Clinical Features with Deep Learning on CBIS-DDSM},
  journal = {Computers in Biology and Medicine},
  year    = {2026},
  note    = {Under review},
  url     = {https://github.com/aliaht99/MammoAI}
}
```

**Dataset:**
```bibtex
@article{lee2017cbisddsm,
  author  = {Lee, Rebecca Sawyer and Gimenez, Francisco and Hoogi, Assaf and Rubin, Daniel},
  title   = {Curated Breast Imaging Subset of DDSM (CBIS-DDSM)},
  journal = {The Cancer Imaging Archive},
  year    = {2017},
  doi     = {10.7937/K9/TCIA.2016.7O02S9CY}
}
```

---

## License

MIT — see [LICENSE](LICENSE)

## Author

**Ali Hamza** — MSc Advanced Engineering Management, Leeds Beckett University  
alihamza.aht.99@gmail.com | [github.com/aliaht99](https://github.com/aliaht99)

## Acknowledgements

- Dataset: [TCIA CBIS-DDSM](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)
- EfficientNet: Tan & Le, ICML 2019
- SHAP: Lundberg & Lee, NeurIPS 2017
- GradCAM: Selvaraju et al., ICCV 2017
