import os
import pydicom
import numpy as np
from glob import glob

def get_dicom_series(directory):
    """Return sorted list of DICOM file paths in a directory."""
    dicom_files = sorted(glob(os.path.join(directory, '*.dcm')))
    return dicom_files

def load_dicom_volume(dicom_dir):
    """Load a DICOM series from a directory and return a 3D numpy array."""
    files = get_dicom_series(dicom_dir)
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices], axis=0)
    return volume

def find_chaos_cases(base_dir):
    """Find all CHAOS MR cases and their DICOM series."""
    cases = []
    for case_id in os.listdir(base_dir):
        case_path = os.path.join(base_dir, case_id)
        if not os.path.isdir(case_path):
            continue
        for seq in ['T1DUAL', 'T2SPIR']:
            seq_path = os.path.join(case_path, seq, 'DICOM_anon')
            if os.path.exists(seq_path):
                for phase in os.listdir(seq_path):
                    phase_path = os.path.join(seq_path, phase)
                    if os.path.isdir(phase_path):
                        cases.append({'case': case_id, 'sequence': seq, 'phase': phase, 'path': phase_path})
    return cases
