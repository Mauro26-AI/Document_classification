# 📄 Document Image Classification with EfficientNetB0 — RVL-CDIP Mini

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-datasets-FFD21E?logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Transfer%20Learning-EfficientNetB0-4CAF50" />
  <img src="https://img.shields.io/badge/Task-16--Class%20Classification-blueviolet" />
</p>

> **Author:** [Mauro26-AI](https://github.com/Mauro26-AI)  
> **Reference Paper:** Harley et al., *Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval*, ICDAR 2015  
> **Dataset:** [`dvgodoy/rvl_cdip_mini`](https://huggingface.co/datasets/dvgodoy/rvl_cdip_mini) (HuggingFace Hub)

---

## 📌 Project Overview

This project tackles **multi-class document image classification** on the RVL-CDIP benchmark, a canonical dataset for Intelligent Document Processing (IDP) research. The goal is to automatically classify scanned document images into **16 distinct categories** — from invoices and advertisements to scientific reports and handwritten notes — using a Convolutional Neural Network trained with **two-phase transfer learning**.

The approach is directly applicable to real-world document automation pipelines such as:
- Automated invoice and receipt routing
- Enterprise archive digitisation and indexing
- RPA pipelines for email/document triage
- Legal-tech and compliance document classification

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Source | `dvgodoy/rvl_cdip_mini` (HuggingFace) |
| Original Benchmark | RVL-CDIP (Harley et al., ICDAR 2015) |
| Training samples | 3,200 (200 per class) |
| Validation samples | 400 (25 per class) |
| Test samples | 400 (25 per class) |
| Number of classes | 16 |
| Image format | Grayscale TIFF (variable resolution) |
| Class balance | Perfectly balanced (200 samples/class) |

The **RVL-CDIP Mini** is a curated, class-balanced subset of the full RVL-CDIP corpus (400,000 images). It is ideal for rapid prototyping and architectural experimentation while remaining representative of real-world document distribution.

### 16 Document Categories

```
advertisement    budget           email           file folder
form             handwritten      invoice         letter
memo             news article     presentation    questionnaire
resume           scientific publication    scientific report    specification
```

---

## 🏗️ Architecture & Methodology

### Model: EfficientNetB0 with Custom Classification Head

EfficientNetB0 was chosen for its excellent **accuracy-to-parameter trade-off**, making it particularly well-suited for small datasets (~3,200 training samples) where larger architectures would overfit. Pre-trained on ImageNet, its convolutional backbone already encodes general visual features (edges, textures, shapes) that transfer effectively to document images.

```
EfficientNetB0 Backbone (ImageNet weights)
    └── features (frozen in Phase 1, unfrozen in Phase 2)
Custom Classification Head:
    └── Dropout(0.4)
    └── Linear(1280 → 512)
    └── ReLU
    └── Dropout(0.3)
    └── Linear(512 → 16)
```

**Total parameters:** ~4.67M  
**Trainable (Phase 1):** ~664K (14.2%)  
**Trainable (Phase 2):** ~4.67M (100%)

---

### Training Strategy: Two-Phase Transfer Learning

The training is split into two distinct phases to prevent gradient interference between the randomly-initialised head and the pre-trained backbone.

```
Phase 1 — Frozen Backbone (20 epochs, lr=1e-3)
    ├── Only the custom head is trained
    ├── Backbone weights are locked (frozen)
    └── Goal: stabilise the head before full fine-tuning

                        ↓

Phase 2 — Full Fine-tuning (10 epochs, lr=1e-4)
    ├── All parameters unfrozen
    ├── Very low learning rate to avoid catastrophic forgetting
    └── Goal: specialise the backbone on document features
```

---

## 📁 Repository Structure

```
rvl-cdip-document-classification/
│
├── rvl_cdip_document_classification.ipynb   # Main notebook (full pipeline)
├── rvl_cdip_efficientnet_b0.pt              # Saved model checkpoint (weights + metadata)
├── requirements.txt                          # Pinned dependencies
└── README.md                                 # This file
```

---

## ⚙️ Setup & Reproducibility

### Requirements

```bash
python >= 3.12
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook rvl_cdip_document_classification.ipynb
```

The dataset is automatically downloaded from HuggingFace Hub on first run — no manual download required.

### Reproducibility

All random seeds are fixed (`SEED = 42`) across NumPy, Python's `random`, and PyTorch (including CUDA). `torch.backends.cudnn.deterministic = True` is set for full determinism on GPU.

---

## 🔮 Future Work & Improvements

| Direction | Description | Expected Impact |
|---|---|---|
| **Full RVL-CDIP dataset** | Use `aharley/rvl_cdip` (400K images) | Significant accuracy gain |
| **Vision Transformers** | ViT / DiT (e.g. `microsoft/dit-base-finetuned-rvlcdip`) | SOTA on this benchmark |
| **Multi-modal fusion** | Combine visual features with OCR tokens (LayoutLM) | Better on text-heavy classes |
| **Test-Time Augmentation** | Average predictions over multiple crops/flips | +1-2% robustness |
| **Grad-CAM visualisation** | Inspect which document regions drive predictions | Interpretability |
| **Weighted loss** | Handle class imbalance in the full dataset | Improved minority-class recall |

---

## 🔗 References

- Harley, A. W., Ufkes, A., & Siddiqi, K. (2015). *Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval*. ICDAR 2015. [arXiv:1502.07058](https://arxiv.org/abs/1502.07058)
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- [`dvgodoy/rvl_cdip_mini`](https://huggingface.co/datasets/dvgodoy/rvl_cdip_mini) — HuggingFace Dataset Card
- [`microsoft/dit-base-finetuned-rvlcdip`](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip) — Current SOTA model on RVL-CDIP

---

## 📄 License

This project is released under the **MIT License**. The RVL-CDIP dataset is provided for research purposes — refer to the original paper and HuggingFace dataset card for usage terms.
