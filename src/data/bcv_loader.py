import os
import nibabel as nib
import numpy as np

def get_bcv_image_label_pairs(images_dir, labels_dir):
    """
    Returns a list of (image_path, label_path) tuples for BCV dataset.
    Assumes filenames match between images and labels.
    """
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
    pairs = []
    for img in image_files:
        if img in label_files:
            pairs.append((os.path.join(images_dir, img), os.path.join(labels_dir, img)))
    return pairs

def load_bcv_case(image_path, label_path):
    """
    Loads a single BCV case (image and label) as numpy arrays.
    """
    image = nib.load(image_path).get_fdata().astype(np.float32)
    label = nib.load(label_path).get_fdata().astype(np.uint8)
    return image, label
