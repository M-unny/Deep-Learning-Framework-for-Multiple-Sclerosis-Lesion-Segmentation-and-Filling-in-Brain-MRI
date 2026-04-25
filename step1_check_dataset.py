# STEP 1 - Test that yo-r dataset loads correctly
# R-n this first to confirm yo-r MRI files are fo-nd

import os
import glob

DATA_DIR = r"C:\Users\Pujith\OneDrive\Desktop\ms in brian\training"

print("=" * 40)
print("STEP 1 - Checking your dataset folder")
print("=" * 40)
print("")
print("Looking in: " + DATA_DIR)
print("")

if not os.path.isdir(DATA_DIR):
    print("ERROR: Folder not found!")
    print("Check that DATA_DIR path is correct.")
else:
    patients = sorted(os.listdir(DATA_DIR))
    print("Patient folders found: " + str(len(patients)))
    print("")

    for patient in patients:
        patient_path = os.path.join(DATA_DIR, patient)
        if not os.path.isdir(patient_path):
            continue

        mri_files  = glob.glob(os.path.join(patient_path, "preprocessed", "*.nii*"))
        mask_files = glob.glob(os.path.join(patient_path, "masks", "*.nii*"))

        print("Patient: " + patient)

        if mri_files:
            print("  MRI  : " + mri_files[-1])
        else:
            print("  MRI  : NOT FOUND")

        if mask_files:
            print("  Mask : " + mask_files[-1])
        else:
            print("  Mask : NOT FOUND")

        print("")

    print("=" * 40)
    print("If all patients show MRI and Mask paths above,")
    print("your dataset is ready. Move on to step2_load_and_slice.py")
    print("=" * 40)
