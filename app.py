from flask import Flask, request, render_template, send_file, jsonify
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
import io
import base64
from werkzeug.utils import secure_filename
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER  = "uploads"
RESULT_FOLDER  = "results"
SEG_CHECKPOINT = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\seg_best1.pth"
FILL_CHECKPOINT= r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\best 1.pth"
IMG_SIZE       = 256
BATCH_SIZE     = 4
SEG_THRESHOLD  = 0.5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

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

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model  = None
fill_model = None

def load_models():
    global seg_model, fill_model
    ok = True
    if os.path.isfile(SEG_CHECKPOINT):
        ckpt = torch.load(SEG_CHECKPOINT, map_location=device)
        seg_model = UNet().to(device)
        seg_model.load_state_dict(ckpt["model_state"])
        seg_model.eval()
        print("Segmentation model loaded from epoch: " + str(ckpt.get("epoch", "?")))
    else:
        print("WARNING: seg_best.pth not found")
        ok = False
    if os.path.isfile(FILL_CHECKPOINT):
        ckpt = torch.load(FILL_CHECKPOINT, map_location=device)
        fill_model = UNet().to(device)
        fill_model.load_state_dict(ckpt["model_state"])
        fill_model.eval()
        print("Filling model loaded from epoch: " + str(ckpt.get("epoch", "?")))
    else:
        print("WARNING: best.pth not found")
        ok = False
    return ok

def run_unet(model, norm_vol):
    n_slices = norm_vol.shape[2]
    H_orig   = norm_vol.shape[0]
    W_orig   = norm_vol.shape[1]
    tensors  = []
    for i in range(n_slices):
        t = torch.from_numpy(norm_vol[:, :, i]).unsqueeze(0).float()
        t = F.interpolate(t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                          mode="bilinear", align_corners=False).squeeze(0)
        tensors.append(t)
    batch    = torch.stack(tensors)
    all_pred = []
    loader   = DataLoader(TensorDataset(batch), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (b,) in loader:
            p = model(b.to(device))
            p = F.interpolate(p, size=(H_orig, W_orig),
                              mode="bilinear", align_corners=False)
            all_pred.append(p.cpu())
    pred_tensor = torch.cat(all_pred, dim=0)
    return np.stack([pred_tensor[i, 0].numpy() for i in range(n_slices)], axis=2)

def make_b64_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def get_best_lesion_slices(mask_vol, n=6):
    """Get slices with most lesion pixels."""
    counts = []
    for i in range(mask_vol.shape[2]):
        counts.append((mask_vol[:, :, i].sum(), i))
    counts.sort(reverse=True)
    return [idx for _, idx in counts[:n] if _ > 0]

def make_segmentation_image(norm_vol, mask_vol):
    """
    Shows lesion slices with red overlay on brain scan.
    Evaluator can clearly see WHERE lesions are.
    """
    lesion_slices = get_best_lesion_slices(mask_vol, n=6)
    if not lesion_slices:
        lesion_slices = [mask_vol.shape[2] // 2]

    n    = len(lesion_slices)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3.5))
    fig.patch.set_facecolor("black")
    if n == 1:
        axes = [axes]

    fig.text(0.5, 1.02, "STEP 1 - Segmentation: Lesions Found (shown in red)",
             color="white", fontsize=12, fontweight="bold", ha="center")

    for ax, sl_idx in zip(axes, lesion_slices):
        mri_sl  = norm_vol[:, :, sl_idx]
        mask_sl = mask_vol[:, :, sl_idx]

        # Show MRI in grey
        ax.imshow(mri_sl.T, cmap="gray", origin="lower",
                  vmin=norm_vol.min(), vmax=norm_vol.max())

        # Overlay lesion mask in red
        red_overlay = np.zeros((*mri_sl.T.shape, 4))
        red_overlay[mask_sl.T > 0] = [1, 0, 0, 0.6]  # red with 60% opacity
        ax.imshow(red_overlay, origin="lower")

        ax.set_title("Slice " + str(sl_idx), color="yellow",
                     fontsize=9, fontfamily="monospace")
        ax.axis("off")

    plt.tight_layout(pad=0.3)
    return make_b64_image(fig)

def make_comparison_image(norm_vol, mask_vol, filled_blend):
    """
    Shows 3 columns for lesion slices:
    Original MRI | Lesion Highlighted | Filled Output
    """
    lesion_slices = get_best_lesion_slices(mask_vol, n=4)
    if not lesion_slices:
        lesion_slices = [mask_vol.shape[2] // 2]

    n = len(lesion_slices)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3))
    fig.patch.set_facecolor("black")
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(["Original MRI", "Lesions Found (red)", "After Filling"]):
        axes[0, col].set_title(title, color="white", fontsize=11,
                               fontweight="bold", pad=10)

    vmin = norm_vol.min()
    vmax = norm_vol.max()

    for row, sl_idx in enumerate(lesion_slices):
        mri_sl    = norm_vol[:, :, sl_idx]
        mask_sl   = mask_vol[:, :, sl_idx]
        filled_sl = filled_blend[:, :, sl_idx]

        # Column 1: Original MRI
        axes[row, 0].imshow(mri_sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 0].set_ylabel("Slice " + str(sl_idx), color="#aaa", fontsize=9)

        # Column 2: MRI with lesion overlay in red
        axes[row, 1].imshow(mri_sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        red_overlay = np.zeros((*mri_sl.T.shape, 4))
        red_overlay[mask_sl.T > 0] = [1, 0.2, 0.2, 0.65]
        axes[row, 1].imshow(red_overlay, origin="lower")

        # Column 3: Filled output
        axes[row, 2].imshow(filled_sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

        for col in range(3):
            axes[row, col].set_facecolor("black")
            axes[row, col].axis("off")

    plt.suptitle("Segmentation + Filling Pipeline Results",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.4)
    return make_b64_image(fig)

def make_axial_grid(filled_blend):
    """20 axial slices of the filled MRI."""
    n_slices     = filled_blend.shape[2]
    brain_slices = [i for i in range(n_slices) if filled_blend[:, :, i].mean() > 0.05]
    step  = max(1, len(brain_slices) // 20)
    picks = brain_slices[::step][:20]
    vmin  = filled_blend.min()
    vmax  = filled_blend.max()

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.patch.set_facecolor("black")
    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor("black")
        if idx < len(picks):
            sl = filled_blend[:, :, picks[idx]]
            ax.imshow(sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax.text(4, 6, "Ax " + str(picks[idx]),
                    color="yellow", fontsize=7, fontfamily="monospace")
        ax.axis("off")
    plt.suptitle("Filled MRI - Axial Slices",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.3)
    return make_b64_image(fig)

def make_3plane_view(filled_blend):
    """Axial, coronal, sagittal centre slices."""
    n_slices = filled_blend.shape[2]
    cx = filled_blend.shape[0] // 2
    cy = filled_blend.shape[1] // 2
    cz = filled_blend.shape[2] // 2
    vmin = filled_blend.min()
    vmax = filled_blend.max()

    best_z, best_mean = cz, 0
    for z in range(max(0, cz - 20), min(n_slices, cz + 20)):
        m = filled_blend[:, :, z].mean()
        if m > best_mean:
            best_mean = m
            best_z = z

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("black")
    for ax, (sl, label) in zip(axes, [
        (filled_blend[:, :, best_z], "AXIAL"),
        (filled_blend[:, cy, :],     "CORONAL"),
        (filled_blend[cx, :, :],     "SAGITTAL"),
    ]):
        ax.set_facecolor("black")
        ax.imshow(sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(label, color="white", fontsize=12, fontweight="bold", pad=10)
        h, w = sl.T.shape
        ax.axhline(h // 2, color="cyan", linewidth=0.5, alpha=0.4)
        ax.axvline(w // 2, color="cyan", linewidth=0.5, alpha=0.4)
        ax.axis("off")
    plt.suptitle("Filled MRI - 3 Plane View",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=1.0)
    return make_b64_image(fig)

def run_full_pipeline(mri_path):
    raw_vol            = nib.load(mri_path).get_fdata(dtype=np.float32)
    orig_min, orig_max = raw_vol.min(), raw_vol.max()
    norm_vol           = (raw_vol - orig_min) / (orig_max - orig_min)
    n_slices           = norm_vol.shape[2]
    H_orig             = norm_vol.shape[0]
    W_orig             = norm_vol.shape[1]
    ref                = nib.load(mri_path)

    # Step 1: Segmentation
    seg_prob      = run_unet(seg_model, norm_vol)
    mask_vol      = (seg_prob >= SEG_THRESHOLD).astype(np.float32)
    lesion_voxels = int(mask_vol.sum())

    # Step 2: Prepare masked slices
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

    # Step 3: Filling
    all_preds = []
    loader    = DataLoader(TensorDataset(masked_batch), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            preds = fill_model(batch.to(device))
            preds = F.interpolate(preds, size=(H_orig, W_orig),
                                  mode="bilinear", align_corners=False)
            all_preds.append(preds.cpu())
    pred_tensor  = torch.cat(all_preds, dim=0)
    pred_slices  = [pred_tensor[i, 0].numpy() for i in range(n_slices)]
    filled_norm  = np.stack(pred_slices, axis=2)
    filled_blend = norm_vol.copy()
    filled_blend[mask_vol > 0] = filled_norm[mask_vol > 0]
    filled_final = filled_blend * (orig_max - orig_min) + orig_min

    # Save NIfTI
    output_path = os.path.join(RESULT_FOLDER, "filled_mri.nii.gz")
    nib.save(nib.Nifti1Image(filled_final.astype(np.float32),
                              ref.affine, ref.header), output_path)

    # Metrics
    all_psnr, all_ssim, all_mse = [], [], []
    for i in range(n_slices):
        if mask_vol[:, :, i].sum() < 1:
            continue
        o = norm_vol[:, :, i].astype(np.float64)
        f = filled_blend[:, :, i].astype(np.float64)
        all_mse.append(float(np.mean((o - f) ** 2)))
        all_psnr.append(float(skimage_psnr(o, f, data_range=1.0)))
        all_ssim.append(float(skimage_ssim(o, f, data_range=1.0)))

    metrics = {
        "mse":           round(float(np.mean(all_mse))  if all_mse  else 0, 6),
        "psnr":          round(float(np.mean(all_psnr)) if all_psnr else 0, 2),
        "ssim":          round(float(np.mean(all_ssim)) if all_ssim else 0, 4),
        "lesion_voxels": lesion_voxels,
        "mri_shape":     list(norm_vol.shape),
    }

    # Generate all 4 images
    seg_b64        = make_segmentation_image(norm_vol, mask_vol)
    comparison_b64 = make_comparison_image(norm_vol, mask_vol, filled_blend)
    axial_b64      = make_axial_grid(filled_blend)
    plane_b64      = make_3plane_view(filled_blend)

    return metrics, seg_b64, comparison_b64, axial_b64, plane_b64

@app.route("/")
def index():
    return render_template("index.html",
                           seg_loaded=seg_model  is not None,
                           fill_loaded=fill_model is not None)

@app.route("/predict", methods=["POST"])
def predict():
    if seg_model is None or fill_model is None:
        return jsonify({"error": "One or both models not loaded."}), 500
    if "mri_file" not in request.files:
        return jsonify({"error": "Please upload an MRI file."}), 400
    mri_file = request.files["mri_file"]
    if mri_file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    mri_path = os.path.join(UPLOAD_FOLDER, secure_filename(mri_file.filename))
    mri_file.save(mri_path)
    try:
        metrics, seg_b64, comparison_b64, axial_b64, plane_b64 = run_full_pipeline(mri_path)
        return jsonify({
            "success":        True,
            "metrics":        metrics,
            "seg_view":       seg_b64,
            "comparison":     comparison_b64,
            "axial_view":     axial_b64,
            "plane_view":     plane_b64,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download")
def download():
    path = os.path.join(RESULT_FOLDER, "filled_mri.nii.gz")
    if not os.path.isfile(path):
        return "No result file found.", 404
    return send_file(path, as_attachment=True, download_name="filled_mri.nii.gz")

if __name__ == "__main__":
    print("Loading models...")
    load_models()
    print("Starting Flask server at http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)