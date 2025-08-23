# Dataset Summary for "Show and Segment" Paper

## Training Datasets (12 datasets)

7. **M&Ms** (`campello2021multi`) - Multi-center Cardiac Segmentation
   - Modality: Cardiac MRI
   - 320 cases
   - 3 structures: LV, RV, myocardium

8. **Brain Aging** (`rodrigue2012beta`) - Brain Tissue Segmentation
   - Modality: T1 MRI
   - 213 scans
   - 3 tissue types: CSF, gray matter, white matter

9. **AutoPET** (`gatidis2022whole`) - Whole-body FDG-PET/CT
   - Modalities: PET/CT
   - 1014 studies
   - Tumor lesions

11. **M&Ms RV** (`martin2023deep`) - Right Ventricle Segmentation
    - Modality: Cardiac MRI
    - Extension of M&Ms dataset

12. Additional training dataset mentioned but not specified

## Held-out Test Datasets (5 datasets)

### For OOD Generalization:
1. **ACDC** (`bernard2018deep`) - Automated Cardiac Diagnosis
   - Modality: Cardiac MRI
   - 100 cases
   - Cross-center variation test

2. **SegTHOR** (`lambert2020segthor`) - Thoracic Organs at Risk
   - Modality: CT
   - 40 cases
   - 4 structures: heart, aorta, trachea, esophagus

## Key Implementation References

- **nnUNet** (`isensee2021nnu`) - Baseline comparison method
- **Lamb Optimizer** (`you2019large`) - Training optimization
- **Visual In-context Learning** (`zhang2023makes`) - Inference strategy analysis