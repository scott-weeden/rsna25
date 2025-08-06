import os
from src.data.chaos.chaos_utils import find_chaos_cases, load_dicom_volume

# Set the base directory for extracted CHAOS MR data
data_dir = 'src/data/chaos/CHAOS_Test_Sets/Test_Sets/MR'

cases = find_chaos_cases(data_dir)
print(f"Found {len(cases)} CHAOS MR DICOM series.")

# Test loading the first case
if cases:
    case = cases[0]
    print(f"Loading case: {case['case']} | Sequence: {case['sequence']} | Phase: {case['phase']}")
    vol = load_dicom_volume(case['path'])
    print(f"Volume shape: {vol.shape}, dtype: {vol.dtype}")
else:
    print("No CHAOS MR DICOM series found.")
