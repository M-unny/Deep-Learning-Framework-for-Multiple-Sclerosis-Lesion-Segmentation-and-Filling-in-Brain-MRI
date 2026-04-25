# STEP 3 - Create masked inp-ts (zero o-t lesion regions)
# This shows what the model will receive as INPUT d-ring training

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------ EDIT THESE PATHS ------------------------------------------------------------------------------------------------------------------
MRI_PATH  = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\training\training01\preprocessed\training01_01_mprage_pp.nii"
MASK_PATH = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\training\training01\masks\training01_01_mask1.nii"
OUT_DIR   = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\step3_output"
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 40)
print("STEP 3 - Create masked inputs for the model")
print("=" * 40)
print("")

# Load and normalise
try:
    mri_vol  = nib.load(MRI_PATH).get_fdata(dtype=np.float32)
except TypeError:
    mri_vol  = nib.load(MRI_PATH).get_fdata()
try:
    mask_vol = nib.load(MASK_PATH).get_fdata(dtype=np.float32)
except TypeError:
    mask_vol = nib.load(MASK_PATH).get_fdata()
vmin, vmax = mri_vol.min(), mri_vol.max()
mri_norm = (mri_vol - vmin) / (vmax - vmin)

# Find lesion slices
 # Find lesion slices
n_slices = mri_vol.shape[2]
lesion_slices = [i for i in range(n_slices) if mask_vol[:, :, i].sum() > 10]
print("Total lesion slices: " + str(len(lesion_slices)))
print("")

# For each lesion slice, create masked version
# Masked = original MRI b-t lesion pixels set to -
print("Creating masked inputs...")
sample = lesion_slices[:6]

fig, axes = plt.subplots(3, len(sample), figsize=(len(sample) * 3, 9))

for col, sl_idx in enumerate(sample):
    original = mri_norm[:, :, sl_idx]       # original MRI slice
    mask_sl  = mask_vol[:, :, sl_idx]       # lesion mask
    masked   = original.copy()
    masked[mask_sl > 0] = 0.0               # zero out lesion region

    vlo, vhi = original.min(), original.max()

    # Row 0: Original MRI
    axes[0, col].imshow(original.T, cmap="gray", origin="lower", vmin=vlo, vmax=vhi)
    axes[0, col].set_title("Slice " + str(sl_idx), fontsize=9)
    axes[0, col].axis("off")

    # Row 1: Masked input (what model sees)
    axes[1, col].imshow(masked.T, cmap="gray", origin="lower", vmin=vlo, vmax=vhi)
    axes[1, col].axis("off")

    # Row 2: Lesion mask
    axes[2, col].imshow(mask_sl.T, cmap="hot", origin="lower")
    axes[2, col].axis("off")

axes[0, 0].set_ylabel("Original MRI", fontsize=10)
axes[1, 0].set_ylabel("Masked Input (lesion = black)", fontsize=10)
axes[2, 0].set_ylabel("Lesion Mask", fontsize=10)

plt.suptitle("STEP 3 - Original vs Masked Input (what U-Net receives)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "step3_masked_inputs.png")
fig.savefig(out_path, dpi=100, bbox_inches="tight")
plt.close(fig)

print("Figure saved -> " + out_path)
print("")
print("  Row 1 = Original MRI slice (TARGET - what model should output)")
print("  Row 2 = Masked input       (INPUT  - what model receives)")
print("  Row 3 = Lesion mask        (shows WHERE the lesion is)")
print("")
print("The model learns to go from Row 2 -> Row 1")
print("")
print("=" * 40)
print("STEP 3 COMPLETE - Move on to step4_build_model.py")
print("=" * 40)
