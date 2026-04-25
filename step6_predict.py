# STEP 6 - Predict and reconstruct filled 3D MRI
# Output: radiologist style MRI viewer (axial, coronal, sagittal)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

# ============================================================
# SETTINGS
# ============================================================
MRI_PATH    = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\training\training01\preprocessed\training01_04_mprage_pp.nii"
MASK_PATH   = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\training\training01\masks\training01_04_mask1.nii"
CHECKPOINT  = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\best.pth"
OUTPUT_PATH = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\results\filled_mri.nii.gz"
OUT_DIR     = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\results"
IMG_SIZE    = 256
BATCH_SIZE  = 4
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# U-NET MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        f = 64
        self.inc   = DoubleConv(1,   f)
        self.down1 = Down(f,   f*2)
        self.down2 = Down(f*2, f*4)
        self.down3 = Down(f*4, f*8)
        self.down4 = Down(f*8, f*16)
        self.up1   = Up(f*16,  f*8)
        self.up2   = Up(f*8,   f*4)
        self.up3   = Up(f*4,   f*2)
        self.up4   = Up(f*2,   f)
        self.outc  = nn.Conv2d(f, 1, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return torch.sigmoid(self.outc(x))

# ============================================================
# SAVE RADIOLOGIST STYLE OUTPUT
# Shows a grid of axial slices like a real MRI report
# Plus 3-plane view (axial, coronal, sagittal)
# ============================================================

def save_mri_viewer(filled_vol, out_dir):
    """
    Saves two images:
    1. mri_report.png  - grid of axial slices like a real MRI report
    2. mri_3plane.png  - axial + coronal + sagittal centre slices
    """

    # -- 1. MRI Report style - grid of axial slices ----------
    n_slices   = filled_vol.shape[2]

    # Find slices that have actual brain (not background)
    brain_slices = []
    for i in range(n_slices):
        sl = filled_vol[:, :, i]
        if sl.mean() > 0.05:   # has brain tissue
            brain_slices.append(i)

    # Pick evenly spaced slices across the brain
    n_show = 20
    step   = max(1, len(brain_slices) // n_show)
    picks  = brain_slices[::step][:n_show]

    # Layout: 4 rows x 5 columns = 20 slices
    rows, cols = 4, 5
    fig, axes  = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.patch.set_facecolor("black")

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor("black")
        if idx < len(picks):
            sl_idx = picks[idx]
            sl     = filled_vol[:, :, sl_idx]
            ax.imshow(sl.T, cmap="gray", origin="lower",
                      vmin=filled_vol.min(), vmax=filled_vol.max())
            ax.text(4, 6, "Ax " + str(sl_idx),
                    color="yellow", fontsize=7,
                    fontfamily="monospace")
        ax.axis("off")

    plt.suptitle("MS Lesion Filling - Axial Slices (Filled MRI)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.3)
    report_path = os.path.join(out_dir, "mri_report.png")
    fig.savefig(report_path, dpi=150, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)
    print("  MRI report saved  -> " + report_path)

    # -- 2. 3-plane view (centre slices) ----------------------
    cx = filled_vol.shape[0] // 2
    cy = filled_vol.shape[1] // 2
    cz = filled_vol.shape[2] // 2

    # Find best axial slice (most brain content near centre)
    best_z = cz
    best_mean = 0
    for z in range(max(0, cz-20), min(n_slices, cz+20)):
        m = filled_vol[:, :, z].mean()
        if m > best_mean:
            best_mean = m
            best_z    = z

    axial_sl    = filled_vol[:, :, best_z]
    coronal_sl  = filled_vol[:, cy, :]
    sagittal_sl = filled_vol[cx, :, :]

    vmin = filled_vol.min()
    vmax = filled_vol.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("black")

    views = [
        (axial_sl,    "AXIAL"),
        (coronal_sl,  "CORONAL"),
        (sagittal_sl, "SAGITTAL"),
    ]

    for ax, (sl, label) in zip(axes, views):
        ax.set_facecolor("black")
        ax.imshow(sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(label, color="white", fontsize=12,
                     fontweight="bold", fontfamily="monospace", pad=10)

        # Add crosshair lines like a real MRI viewer
        h, w = sl.T.shape
        ax.axhline(h // 2, color="cyan", linewidth=0.5, alpha=0.4)
        ax.axvline(w // 2, color="cyan", linewidth=0.5, alpha=0.4)
        ax.axis("off")

    plt.suptitle("MS Lesion Filling - 3 Plane View (Filled MRI)",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=1.0)
    plane_path = os.path.join(out_dir, "mri_3plane.png")
    fig.savefig(plane_path, dpi=150, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)
    print("  3-plane view saved -> " + plane_path)

    return report_path, plane_path


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("MS LESION FILLING - PREDICTION")
print("=" * 60)
print("")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : " + str(device))
print("")

# STEP 1 - Load MRI and mask
print("[1] Loading MRI and mask...")
raw_vol        = nib.load(MRI_PATH).get_fdata(dtype=np.float32)
orig_min, orig_max = raw_vol.min(), raw_vol.max()
norm_vol       = (raw_vol - orig_min) / (orig_max - orig_min)
mask_vol       = nib.load(MASK_PATH).get_fdata(dtype=np.float32)
n_slices       = norm_vol.shape[2]
H_orig         = norm_vol.shape[0]
W_orig         = norm_vol.shape[1]
print("  MRI shape     : " + str(norm_vol.shape))
print("  Lesion voxels : " + str(int(mask_vol.sum())))
print("")

# STEP 2 - Prepare masked slices
print("[2] Preparing masked slices...")
masked_tensors = []
for i in range(n_slices):
    mri_sl  = norm_vol[:, :, i]
    mask_sl = mask_vol[:, :, i]
    masked  = mri_sl.copy()
    masked[mask_sl > 0] = 0.0
    t = torch.from_numpy(masked).unsqueeze(0).float()
    t = F.interpolate(t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                      mode="bilinear", align_corners=False).squeeze(0)
    masked_tensors.append(t)
masked_batch = torch.stack(masked_tensors)
print("  Prepared " + str(n_slices) + " slices")
print("")

# STEP 3 - Load model
print("[3] Loading trained model...")
if not os.path.isfile(CHECKPOINT):
    print("ERROR: best.pth not found at: " + CHECKPOINT)
    raise SystemExit(1)

ckpt  = torch.load(CHECKPOINT, map_location=device)
model = UNet().to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("  Loaded from epoch : " + str(ckpt.get("epoch", "?")))
print("  PSNR              : %.2f dB" % ckpt.get("psnr", float("nan")))
print("  SSIM              : %.4f"    % ckpt.get("ssim", float("nan")))
print("")

# STEP 4 - Run inference
print("[4] Running U-Net inference...")
all_preds = []
loader    = DataLoader(TensorDataset(masked_batch),
                       batch_size=BATCH_SIZE, shuffle=False)
with torch.no_grad():
    for i, (batch,) in enumerate(loader):
        preds = model(batch.to(device))
        preds = F.interpolate(preds, size=(H_orig, W_orig),
                              mode="bilinear", align_corners=False)
        all_preds.append(preds.cpu())
        print("  Batch " + str(i+1) + "/" + str(len(loader)) + " done")
pred_tensor = torch.cat(all_preds, dim=0)
print("")

# STEP 5 - Reconstruct 3D volume
print("[5] Reconstructing 3D volume...")
pred_slices  = [pred_tensor[i, 0].numpy() for i in range(n_slices)]
filled_norm  = np.stack(pred_slices, axis=2)
filled_blend = norm_vol.copy()
filled_blend[mask_vol > 0] = filled_norm[mask_vol > 0]
filled_final = filled_blend * (orig_max - orig_min) + orig_min
print("  Shape: " + str(filled_final.shape))
print("")

# STEP 6 - Save filled MRI as NIfTI
print("[6] Saving filled MRI file...")
ref     = nib.load(MRI_PATH)
new_img = nib.Nifti1Image(filled_final.astype(np.float32), ref.affine, ref.header)
nib.save(new_img, OUTPUT_PATH)
print("  Saved -> " + OUTPUT_PATH)
print("")

# STEP 7 - Compute metrics
print("[7] Computing metrics...")
all_mse, all_psnr, all_ssim = [], [], []
for i in range(n_slices):
    if mask_vol[:, :, i].sum() < 1:
        continue
    o = norm_vol[:, :, i].astype(np.float64)
    f = filled_blend[:, :, i].astype(np.float64)
    all_mse.append(float(np.mean((o - f) ** 2)))
    all_psnr.append(float(skimage_psnr(o, f, data_range=1.0)))
    all_ssim.append(float(skimage_ssim(o, f, data_range=1.0)))
print("  MSE  : %.6f" % np.mean(all_mse))
print("  PSNR : %.2f dB" % np.mean(all_psnr))
print("  SSIM : %.4f" % np.mean(all_ssim))
print("")

# STEP 8 - Save radiologist style MRI viewer images
print("[8] Saving MRI viewer output...")
report_path, plane_path = save_mri_viewer(filled_blend, OUT_DIR)
print("")

print("=" * 60)
print("DONE!")
print("")
print("Files saved:")
print("  Filled MRI file : " + OUTPUT_PATH)
print("  MRI report      : " + report_path)
print("  3-plane view    : " + plane_path)
print("")
print("Open mri_report.png  -> see all axial slices like a real MRI")
print("Open mri_3plane.png  -> see axial, coronal, sagittal views")
print("Open filled_mri.nii.gz in ITK-SNAP for full 3D viewing")
print("=" * 60)