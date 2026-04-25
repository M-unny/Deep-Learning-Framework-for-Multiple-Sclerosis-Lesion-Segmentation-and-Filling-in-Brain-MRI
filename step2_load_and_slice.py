# STEP - - Load one MRI file and convert it into -D slices
# R-n this to confirm NIfTI loading and slicing works

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------ EDIT THESE PATHS ------------------------------------------------------------------------------------------------------------------
MRI_PATH  = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\training\training01\preprocessed\training01_01_mprage_pp.nii"
MASK_PATH = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\training\training01\masks\training01_01_mask1.nii"
OUT_DIR   = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\step2_output"
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 40)
print("STEP 2 - Load MRI and convert to 2D slices")
print("=" * 40)
print("")

# Load MRI
print("Loading MRI from: " + MRI_PATH)
mri_img  = nib.load(MRI_PATH)
try:
    mri_vol  = mri_img.get_fdata(dtype=np.float32)
except TypeError:
    mri_vol  = mri_img.get_fdata()
print("  MRI shape    : " + str(mri_vol.shape))
print("  MRI min/max  : %.2f / %.2f" % (mri_vol.min(), mri_vol.max()))
print("")

# Load mask
print("Loading mask from: " + MASK_PATH)
mask_img = nib.load(MASK_PATH)
try:
    mask_vol = mask_img.get_fdata(dtype=np.float32)
except TypeError:
    mask_vol = mask_img.get_fdata()
print("  Mask shape   : " + str(mask_vol.shape))
print("  Lesion voxels: " + str(int(mask_vol.sum())))
print("")

 # Normalise MRI to [0, 1]
vmin, vmax = mri_vol.min(), mri_vol.max()
if vmax != vmin:
    mri_norm   = (mri_vol - vmin) / (vmax - vmin)
else:
    mri_norm = mri_vol
print("MRI normalised to [0, 1]")
print("")

 # Convert 3D volume to 2D slices along axial axis (axis=2)
n_slices   = mri_vol.shape[2]
mri_slices = [mri_norm[:, :, i] for i in range(n_slices)]
print("Total axial slices: " + str(n_slices))

# Find which slices contain lesions
lesion_slices = []
for i in range(n_slices):
    if mask_vol[:, :, i].sum() > 10:
        lesion_slices.append(i)
print("Slices with lesions: " + str(len(lesion_slices)))
print("Lesion slice indices: " + str(lesion_slices[:10]) + " ...")
print("")

# Save a fig-re showing 6 sample slices
fig, axes = plt.subplots(2, 6, figsize=(18, 6))

# Top row: MRI slices
# Bottom row: Mask slices
sample_indices = lesion_slices[:6] if len(lesion_slices) >= 6 else lesion_slices

for col, sl_idx in enumerate(sample_indices):
    axes[0, col].imshow(mri_norm[:, :, sl_idx].T, cmap="gray", origin="lower")
    axes[0, col].set_title("MRI Slice " + str(sl_idx), fontsize=9)
    axes[0, col].axis("off")

    axes[1, col].imshow(mask_vol[:, :, sl_idx].T, cmap="hot", origin="lower")
    axes[1, col].set_title("Mask Slice " + str(sl_idx), fontsize=9)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("MRI", fontsize=10)
axes[1, 0].set_ylabel("Lesion Mask", fontsize=10)

plt.suptitle("STEP 2 - MRI Slices and Lesion Masks", fontsize=13, fontweight="bold")
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "step2_slices.png")
fig.savefig(out_path, dpi=100, bbox_inches="tight")
plt.close(fig)

print("Figure saved -> " + out_path)
print("Open that image to see your MRI slices and lesion masks.")
print("")
print("=" * 40)
print("STEP 2 COMPLETE - Move on to step3_create_masked_inputs.py")
print("=" * 40)
