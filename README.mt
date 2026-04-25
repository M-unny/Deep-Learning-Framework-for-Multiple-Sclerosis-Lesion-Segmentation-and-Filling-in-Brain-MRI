# MS Lesion Filling Pipeline
### Deep Learning-Based Automated FLAIR MRI Lesion Normalisation for Multiple Sclerosis

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Multiple Sclerosis (MS) causes white matter lesions that appear as **hyperintense regions** on FLAIR MRI scans. These lesions interfere with longitudinal brain volumetry and disease progression tracking by introducing artificial intensity variations.

This project presents an **end-to-end automated pipeline** that:
1. **Detects** MS lesions automatically (no manual mask required at inference)
2. **Fills** the lesion regions with realistic healthy tissue appearance
3. **Outputs** a clean, lesion-free NIfTI file ready for downstream analysis

> **Key Novelty:** Works with **FLAIR MRI only** — no T1, T2, or other modality required.

---

## Pipeline Architecture

```
INPUT: FLAIR MRI (.nii / .nii.gz)
         │
         ▼
┌─────────────────────────────┐
│   STAGE 1 — SEGMENTATION    │
│   2D U-Net                  │
│   → Binary lesion mask      │
│   Best Dice: 0.5198         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   STAGE 2 — INPAINTING      │
│   2D Inpainting U-Net       │
│   → Reconstructs healthy    │
│     tissue in lesion region │
│   Best PSNR: 46.77 dB       │
│   Best SSIM: 0.9977         │
└─────────────┬───────────────┘
              │
              ▼
OUTPUT: Lesion-Filled FLAIR MRI (.nii.gz)
        + Axial slice grid
        + 3-plane view (Axial / Coronal / Sagittal)
        + Metrics: PSNR, SSIM, MSE, lesion voxel count
```

---

## Model Comparison Results

Evaluated on **21 unseen test patients (P54–P75)**:

| Model | Seg Dice ↑ | Fill PSNR ↑ | Fill SSIM ↑ |
|---|---|---|---|
| **U-Net** | **0.5198** | **46.77 dB** | **0.9977** |
| UNet++ | 0.4651 | 30.47 dB | 0.9322 |
| ResU-Net | 0.4534 | 29.26 dB | 0.9697 |
| Attention U-Net | 0.4427 | 32.08 dB | 0.9738 |
| SegNet | 0.4308 | 29.26 dB | 0.9221 |

> U-Net achieved the best results on both segmentation and filling quality.

---

## Project Structure

```
ms-lesion-filling/
│
├── training/                          # Kaggle training notebooks
│   ├── kaggle_preprocess.py           # 3D NIfTI → 2D .npy slices
│   ├── kaggle_train_fast.py           # Baseline U-Net seg + fill
│   ├── kaggle_resunet_train.py        # ResU-Net
│   ├── kaggle_unetpp_train.py         # UNet++
│   ├── kaggle_attunet_train.py        # Attention U-Net
│   ├── kaggle_segnet_train.py         # SegNet
│   ├── kaggle_enhanced_unet.py        # Enhanced U-Net (8 improvements)
│   ├── kaggle_unet_crossval.py        # 5-Fold Cross Validation
│   └── kaggle_vnet3d_train.py         # 3D V-Net (requires T4 GPU)
│
├── testing/
│   ├── test_validation.ipynb          # Single model pair evaluation
│   └── test_all_models.ipynb          # All models comparison + charts
│
├── app/
│   └── app_final.py                   # Flask web application
│
├── models/                            # Trained model checkpoints
│   ├── flair_seg_best.pth             # U-Net segmentation
│   ├── flair_fill_best.pth            # U-Net filling
│   ├── resunet_seg_best.pth
│   ├── resunet_fill_best.pth
│   ├── unetpp_seg_best.pth
│   ├── unetpp_fill_best.pth
│   ├── attunet_seg_best.pth
│   ├── attunet_fill_best.pth
│   ├── segnet_seg_best.pth
│   └── segnet_fill_best.pth
│
├── docs/
│   ├── MS_Lesion_Research_Paper.docx  # IEEE-format research paper
│   ├── Major_Project_Validation_Form_Filled.docx
│   └── Project_Hand_Book_Filled.docx
│
└── README.md
```

---

## Dataset

### Training
- **144 patients** — FLAIR MRI + lesion mask pairs
- Structure:
```
flair_mask_dataset/
├── sample_001/
│   ├── flair.nii
│   └── mask.nii
├── sample_002/
│   ├── flair.nii
│   └── mask.nii
...
└── sample_144/
```

### Testing
- **21 patients** — P54 to P75 (completely unseen during training)
```
final_test_dataset/
├── P54/
│   ├── P54_FLAIR.nii.gz
│   └── P54_MASK.nii.gz
...
└── P75/
```

---

## Installation

### Requirements

```bash
pip install torch torchvision torchaudio
pip install nibabel numpy scikit-image scipy flask matplotlib
pip install scikit-learn
```

### Clone & Setup

```bash
git clone https://github.com/your-repo/ms-lesion-filling.git
cd ms-lesion-filling
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Web Application

```bash
# Place model files in the same directory as app_final.py
# flair_seg_best.pth  +  flair_fill_best.pth

python app_final.py
```

Open browser at `http://localhost:5000`

**Steps:**
1. Upload your FLAIR `.nii` or `.nii.gz` file
2. Click **Process**
3. View results: segmentation overlay, before/after comparison, axial grid, 3-plane view
4. Download the lesion-filled NIfTI file

---

### 2. Train from Scratch on Kaggle

**Step 1 — Preprocess**
```python
# Upload kaggle_preprocess.py as a Kaggle notebook cell
# Set DATA_DIR to your dataset path
# Output: /kaggle/working/preprocessed/flair_slices.npy, mask_slices.npy
```

**Step 2 — Train**
```python
# Upload kaggle_train_fast.py
# Output: /kaggle/working/checkpoints/flair_seg_best.pth
#                                      flair_fill_best.pth
```

> **Recommended GPU:** T4 x2 (CUDA 7.5+). P100 is incompatible with current PyTorch for 3D models.

---

### 3. Run Cross-Validation

```bash
# Upload kaggle_unet_crossval.py to Kaggle
# Uses 5-Fold CV across all 144 patients
# Output: /kaggle/working/crossval_seg_best.pth  (best fold model)
#         /kaggle/working/crossval_checkpoints/crossval_summary.json
```

---

### 4. Evaluate on Test Patients

```bash
# Open test_all_models.ipynb in Kaggle or Jupyter
# Set TEST_DIR to your 21 test patients folder
# Outputs: per-patient Dice/PSNR/SSIM table + bar/radar charts
```

---

## Training Details

| Parameter | Value |
|---|---|
| Input size | 256 × 256 (2D axial slices) |
| Batch size | 16 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=6) |
| Seg loss | Dice + BCE |
| Fill loss | MSE + 0.1 × Gradient |
| Early stopping | Patience = 10 |
| Epochs | 50 (baseline) / 80 (cross-val) |
| GPU | Kaggle T4 / P100 |

---

## Preprocessing

All 3D NIfTI volumes are processed as follows:

```
1. Load FLAIR volume with nibabel
2. Min-max normalise to [0, 1]
3. Extract axial slices (all depths)
4. Resize each slice to 256x256
   (bilinear for FLAIR, nearest-neighbour for mask)
   → Fixes multi-scanner resolution mismatch
5. Keep only slices with 10+ lesion pixels
6. Save as .npy for fast Kaggle training
```

---

## Augmentation (Enhanced / Cross-Val Models)

```
- Horizontal flip          (p=0.5)
- Vertical flip            (p=0.5)
- Random 90 degree rotation  (p=0.5)
- Brightness jitter +-15%  (p=0.5)
- Gaussian noise s=0.01    (p=0.5)
- Elastic deformation      (Enhanced U-Net only)
```

---

## Inference Pipeline

```python
# 1. Load FLAIR .nii → normalize [0,1] → extract axial slices
# 2. Segmentation model → probability map per slice
# 3. Threshold at 0.5 → binary lesion mask
# 4. Zero out lesion pixels in each slice
# 5. Filling model → reconstruct full slice
# 6. Blend: output[mask==1] = filled[mask==1]  (healthy tissue unchanged)
# 7. Denormalize → reconstruct 3D volume preserving original affine + header
# 8. Save as .nii.gz
```

---

## Known Issues and Limitations

| Issue | Status | Fix |
|---|---|---|
| P100 CUDA 6.0 incompatible with current PyTorch for 3D models | Known | Use T4 GPU |
| Dice capped ~0.52 with FLAIR-only input | Known | Add T1 channel (planned) |
| Web app runs locally only | Known | Cloud deployment planned |
| 144 patient dataset is small | Known | Seeking additional data |

---

## Roadmap

- [x] Baseline U-Net (FLAIR only, 144 patients)
- [x] 5-model architecture comparison (U-Net, ResU-Net, UNet++, Attention U-Net, SegNet)
- [x] Flask web application with 4-panel visualisation
- [x] 5-Fold cross-validation training
- [x] IEEE research paper
- [ ] T1 + FLAIR dual-channel model (target Dice: 0.65+)
- [ ] 3D V-Net training (T4 GPU required)
- [ ] Patch-based lesion-centred training (target Dice: 0.68+)
- [ ] Cloud deployment (AWS / GCP)
- [ ] Ensemble + Test-Time Augmentation (target Dice: 0.75+)

## Acknowledgements

- Kaggle for GPU compute (T4 / P100)
- PyTorch, nibabel, scikit-image open source libraries
- SRMIST Department of Data Science and Business Systems
