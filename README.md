# RSNA Intracranial Aneurysm Detection Challenge

**Competition Page: [Kaggle - RSNA 2025 Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-2025-intracranial-aneurysm-detection)**

## Overview

This repository contains code and resources for the RSNA 2025 Intracranial Aneurysm Detection Challenge, hosted by the Radiological Society of North America (RSNA) in collaboration with the American Society of Neuroradiology (ASNR), the Society of Neurointerventional Surgery (SNIS), and the European Society of Neuroradiology (ESNR).

### Challenge Goal
Create machine learning models to detect and precisely locate intracranial aneurysms across various types of medical images, including:
- CTA (Computed Tomography Angiography)
- MRA (Magnetic Resonance Angiography)
- T1 post-contrast MRI
- T2-weighted MRI

## Medical Context

Intracranial aneurysms affect approximately 3% of the global population and cause roughly 500,000 deaths annually worldwide. Up to 50% of aneurysms are only diagnosed after rupture, which can result in severe illness or death. Early detection through automated solutions could save lives by enabling timely intervention.

## Timeline

- **July 28, 2025**: Competition Start Date
- **October 7, 2025**: Entry & Team Merger Deadline
- **October 14, 2025**: Final Submission Deadline
- **October 27, 2025**: Winners' Requirement Deadline
- **November 2025**: RSNA Annual Meeting - AI Challenge Recognition Event

All deadlines are at 11:59 PM UTC.

## Evaluation Metric

Submissions are evaluated using a **weighted multilabel Area Under the ROC Curve (AUC-ROC)**:

- **Aneurysm Present** label: Weight = 13
- **Other 13 location labels**: Weight = 1 each

Final Score = Average of (Aneurysm Present score + Average of 13 location scores)

## Prize Distribution

- 🥇 **1st Place**: $12,000
- 🥈 **2nd Place**: $10,000
- 🥉 **3rd Place**: $8,000
- **4th Place**: $5,000
- **5th-9th Places**: $3,000 each

Winners will be invited to the RSNA Annual Meeting AI Challenge Recognition Event with waived fees.

## Repository Structure

```
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── train/
│   ├── test/
│   └── sample_submission.csv
├── notebooks/
│   ├── EDA.ipynb
│   ├── baseline_model.ipynb
│   └── submission_example.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   └── architectures.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── validate.py
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
├── configs/
│   └── default_config.yaml
└── scripts/
    ├── train.py
    ├── predict.py
    └── submit.py
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM recommended
- 100GB+ disk space for data

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/mister-weeden/rsna-aneurysm-detection
cd rsna-aneurysm-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download competition data:
```bash
# Install Kaggle API if not already installed
pip install kaggle

# Download data (requires Kaggle API credentials)
kaggle competitions download -c rsna-2025-intracranial-aneurysm-detection
unzip rsna-2025-intracranial-aneurysm-detection.zip -d data/
```

## Usage

### Data Exploration
```bash
jupyter notebook notebooks/EDA.ipynb
```

### Training a Model
```bash
python scripts/train.py --config configs/default_config.yaml
```

### Making Predictions
```bash
python scripts/predict.py --model_path models/best_model.pth --data_path data/test/
```

### Creating a Submission
```bash
python scripts/submit.py --predictions predictions.csv --output submission.csv
```

## Model Development Guidelines

### Data Considerations
- Handle multi-institutional data variations
- Account for different scanner types and imaging protocols
- Implement robust preprocessing for CTA, MRA, and MRI modalities

### Key Challenges
- Class imbalance (aneurysms are rare)
- Small object detection (aneurysms can be tiny)
- Multi-modal imaging integration
- Generalization across different imaging protocols

### Recommended Approaches
- 3D CNN architectures for volumetric data
- Attention mechanisms for small object detection
- Multi-scale feature extraction
- Ensemble methods combining different modalities

## Submission Requirements

### Code Competition Rules
- **Runtime Limit**: ≤ 12 hours (CPU or GPU)
- **Internet Access**: Disabled during inference
- **External Data**: Freely available public data and pre-trained models allowed
- **Output Format**: `submission.csv`

### Winners' Obligations
1. Open-source code and model weights
2. Short video presentation of approach
3. Method description document
4. (Recommended) Publicly hosted model for testing

## Evaluation Metrics Implementation

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def weighted_multilabel_auc(y_true, y_pred):
    """
    Calculate weighted multilabel AUC-ROC score.
    
    Args:
        y_true: Ground truth labels (n_samples, 14)
        y_pred: Predicted probabilities (n_samples, 14)
    
    Returns:
        Weighted average AUC-ROC score
    """
    # Calculate AUC for each label
    auc_scores = []
    for i in range(14):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        auc_scores.append(auc)
    
    # Apply weights (13 for aneurysm present, 1 for others)
    aneurysm_present_score = auc_scores[0] * 13
    location_scores = sum(auc_scores[1:])
    
    # Final score is average of weighted scores
    final_score = (aneurysm_present_score + location_scores) / 14
    
    return final_score
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Resources

### Official Resources
- [Competition Page](https://www.kaggle.com/competitions/rsna-2025-intracranial-aneurysm-detection)
- [Discussion Forum](https://www.kaggle.com/competitions/rsna-2025-intracranial-aneurysm-detection/discussion)
- [Data Description](https://www.kaggle.com/competitions/rsna-2025-intracranial-aneurysm-detection/data)

### Relevant Papers
- [Deep Learning for Intracranial Aneurysm Detection](https://example.com)
- [Multi-modal Medical Image Analysis](https://example.com)
- [3D CNN Architectures for Medical Imaging](https://example.com)

## Acknowledgments

We thank:
- RSNA, ASNR, SNIS, and ESNR for organizing this challenge
- All data contributing institutions
- The medical professionals who annotated the dataset
- DEEPNOID for supporting the challenge
- MD.ai for providing annotation tools


## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{rsna2025aneurysm,
  title={RSNA 2025 Intracranial Aneurysm Detection Challenge Solution},
  author={Your Team Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/[your-username]/rsna-aneurysm-detection}
}
```

## Contact

For questions or collaborations:
- Email: [your-email@example.com]
- Kaggle: [@your-kaggle-username]
- GitHub Issues: [Create an issue](https://github.com/[your-username]/rsna-aneurysm-detection/issues)

---

**Note**: This repository is part of our submission to the RSNA 2025 Intracranial Aneurysm Detection Challenge. The goal is to develop automated solutions that can accurately detect brain aneurysms across various imaging modalities, ultimately helping to save lives through earlier intervention.
