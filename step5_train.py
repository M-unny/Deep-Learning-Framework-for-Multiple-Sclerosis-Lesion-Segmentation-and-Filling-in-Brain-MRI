# STEP - - Train the U-Net model
# This is the main training step. It will take time (3- min to a few ho-rs).
# A checkpoint is saved after every epoch so yo- never lose progress.

import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

# ==============================================================
# SETTINGS - Edit these
# ==============================================================
DATA_DIR   = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\training"
OUTPUT_DIR = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\checkpoints"
EPOCHS     = 40
BATCH_SIZE = 4
LR         = 1e-4
IMG_SIZE   = 256
# ==============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------ Helper f-nctions ------------------------------------------------------------------------------------------------------------------------------

def load_nii(path):
    try:
        return nib.load(path).get_fdata(dtype=np.float32)
    except TypeError:
        return nib.load(path).get_fdata()

def normalize_volume(volume):
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(volume)
    return (volume - vmin) / (vmax - vmin)

def volume_to_slices(volume, axis=2):
    return [np.take(volume, i, axis=axis) for i in range(volume.shape[axis])]

def create_masked_input(mri_slice, mask_slice):
    masked = mri_slice.copy()
    masked[mask_slice > 0] = 0.0
    return masked

def find_patient_pairs(root_dir):
    pairs = []
    print("Scanning: " + root_dir)
    for patient in sorted(os.listdir(root_dir)):
        patient_path = os.path.join(root_dir, patient)
        if not os.path.isdir(patient_path):
            continue
        mri_files  = glob.glob(os.path.join(patient_path, "preprocessed", "*.nii*"))
        mask_files = glob.glob(os.path.join(patient_path, "masks", "*.nii*"))
        if mri_files and mask_files:
            pairs.append((mri_files[-1], mask_files[-1]))
            print("  [FOUND] " + patient)
        else:
            print("  [SKIP]  " + patient)
    print("Total pairs: " + str(len(pairs)))
    print("")
    return pairs

# ------ Dataset ---------------------------------------------------------------------------------------------------------------------------------------------------------

class MRIDataset(Dataset):
    def __init__(self, patient_pairs, target_size=(256, 256)):
        self.samples = []
        self.resize  = transforms.Resize(
            target_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )
        print("Building dataset...")
        for mri_path, mask_path in patient_pairs:
            print("  Loading: " + os.path.basename(mri_path))
            mri_vol  = normalize_volume(load_nii(mri_path))
            mask_vol = load_nii(mask_path)
            for mri_sl, mask_sl in zip(volume_to_slices(mri_vol),
                                       volume_to_slices(mask_vol)):
                lesion_px = int((mask_sl > 0).sum())
                if lesion_px >= 10:
                    masked = create_masked_input(mri_sl, mask_sl)
                    self.samples.append((masked, mri_sl))
                elif np.random.rand() < 0.1:
                    self.samples.append((mri_sl.copy(), mri_sl))
        print("Dataset ready: " + str(len(self.samples)) + " slices")
        print("")

    def _to_tensor(self, arr):
        t = torch.from_numpy(arr).unsqueeze(0).float()
        return self.resize(t)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        masked, original = self.samples[idx]
        inp = self._to_tensor(masked)
        tgt = self._to_tensor(original)
        if np.random.rand() > 0.5:
            inp = torch.flip(inp, dims=[-1])
            tgt = torch.flip(tgt, dims=[-1])
        return inp, tgt

# ------ U-Net ---------------------------------------------------------------------------------------------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.net(x)

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

# ------ U-Net ------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        f = 64

        self.inc   = DoubleConv(1, f)
        self.down1 = Down(f, f*2)
        self.down2 = Down(f*2, f*4)
        self.down3 = Down(f*4, f*8)
        self.down4 = Down(f*8, f*16)

        self.up1 = Up(f*16, f*8)
        self.up2 = Up(f*8,  f*4)
        self.up3 = Up(f*4,  f*2)
        self.up4 = Up(f*2,  f)

        self.outc = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return torch.sigmoid(self.outc(x))


# ------ Loss -------------------------------------------------------------------

class InpaintingLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):

        loss_mse = self.mse(pred, target)

        pred_dx = pred[:,:,:,1:] - pred[:,:,:,:-1]
        pred_dy = pred[:,:,1:,:] - pred[:,:,:-1,:]

        tgt_dx = target[:,:,:,1:] - target[:,:,:,:-1]
        tgt_dy = target[:,:,1:,:] - target[:,:,:-1,:]

        loss_grad = self.mse(pred_dx, tgt_dx) + self.mse(pred_dy, tgt_dy)

        return loss_mse + 0.1 * loss_grad, loss_mse, loss_grad
   
# ------ Loss ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        loss_mse  = self.mse(pred, target)
        pred_dx   = pred[:,:,:,1:]   - pred[:,:,:,:-1]
        pred_dy   = pred[:,:,1:,:]   - pred[:,:,:-1,:]
        tgt_dx    = target[:,:,:,1:] - target[:,:,:,:-1]
        tgt_dy    = target[:,:,1:,:] - target[:,:,:-1,:]
        loss_grad = self.mse(pred_dx, tgt_dx) + self.mse(pred_dy, tgt_dy)
        return loss_mse + 0.1 * loss_grad, loss_mse, loss_grad

# ------ Metrics ---------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_metrics(target, pred):
    target = target.astype(np.float64)
    pred   = pred.astype(np.float64)
    mse    = float(np.mean((target - pred) ** 2))
    psnr   = float(skimage_psnr(target, pred, data_range=1.0))
    ssim   = float(skimage_ssim(target, pred, data_range=1.0))
    return mse, psnr, ssim

# ------ Training ------------------------------------------------------------------------------------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss, mse, grad = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if i % 20 == 0:
            print("  Epoch %d | Batch %d/%d | Loss=%.4f" % (
                epoch, i, len(loader), loss.item()))
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_psnr, all_ssim = [], []
    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        preds   = model(inputs)
        loss, _, _ = criterion(preds, targets)
        val_loss += loss.item()
        for p, t in zip(preds.cpu().numpy()[:,0], targets.cpu().numpy()[:,0]):
            _, psnr, ssim = compute_metrics(t, p)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
    return val_loss / len(loader), float(np.mean(all_psnr)), float(np.mean(all_ssim))

# ------ Main ------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("=" * 40)
print("STEP 5 - Training the U-Net Model")
print("=" * 40)
print("")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
if device.type == "cpu":
    print("NOTE: Running on CPU - training will be slow.")
    print("      Each epoch may take 10-30 minutes.")
print("")

pairs = find_patient_pairs(DATA_DIR)
if not pairs:
    print("ERROR: No patient pairs found. Check DATA_DIR.")
    sys.exit(1)

full_ds = MRIDataset(pairs, target_size=(IMG_SIZE, IMG_SIZE))
n_val   = max(1, int(len(full_ds) * 0.1))
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
print("Train slices: " + str(n_train))
print("Val slices  : " + str(n_val))
print("")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)

val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

model     = UNet().to(device)
criterion = InpaintingLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("U-Net parameters: " + str(n_params))
print("")
print("Starting training...")
print("-" * 40)

best_val_loss    = float("inf")
patience_counter = 0
PATIENCE         = 10
history          = {"train": [], "val": [], "psnr": [], "ssim": []}

for epoch in range(1, EPOCHS + 1):
    t          = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
    val_loss, psnr, ssim = validate(model, val_loader, criterion, device)
    elapsed    = time.time() - t
    scheduler.step(val_loss)

    history["train"].append(train_loss)
    history["val"].append(val_loss)
    history["psnr"].append(psnr)
    history["ssim"].append(ssim)

    print("")
    print("Epoch %d/%d (%.-fs) | Train=%.-f | Val=%.-f | PSNR=%.-fdB | SSIM=%.4f" % (
        epoch, EPOCHS, elapsed, train_loss, val_loss, psnr, ssim))

    # Save latest checkpoint every epoch
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss":   best_val_loss,
    }, os.path.join(OUTPUT_DIR, "latest.pth"))

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_loss":    best_val_loss,
            "psnr":        psnr,
            "ssim":        ssim,
        }, os.path.join(OUTPUT_DIR, "best.pth"))
        print("  [BEST MODEL SAVED] epoch=" + str(epoch) + " val_loss=%.-f" % best_val_loss)
    else:
        patience_counter += 1
        print("  No improvement (" + str(patience_counter) + "/" + str(PATIENCE) + ")")
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    print("")

# Save training history plot
np.save(os.path.join(OUTPUT_DIR, "history.npy"), history)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
ep = range(1, len(history["train"]) + 1)
ax1.plot(ep, history["train"], label="Train", color="royalblue")
ax1.plot(ep, history["val"],   label="Val",   color="tomato")
ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(ep, history["psnr"], color="darkorange")
ax2.set_title("PSNR (dB)"); ax2.grid(True, alpha=0.3)
ax3.plot(ep, history["ssim"], color="seagreen")
ax3.set_title("SSIM"); ax3.grid(True, alpha=0.3)
plt.suptitle("Training History", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=100, bbox_inches="tight")
plt.close(fig)

print("=" * 40)
print("STEP 5 COMPLETE")
print("Best val loss : %.4f" % best_val_loss)
print("Model saved in: " + OUTPUT_DIR)
print("Now run step6_predict.py to fill lesions in a new MRI")
print("=" * 40)
