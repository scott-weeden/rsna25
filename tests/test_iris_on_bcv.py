"""
Test IRIS model (trained on AMOS) on BCV dataset
Assumes BCV data is in src/data/bcv/imagesTr/ and labelsTr/
"""
import os
import torch
import numpy as np
from src.models.iris_model import IRISModel, IRISInference
from src.data.bcv.bcv_utils import get_bcv_image_label_pairs, load_bcv_case
import nibabel as nib

def main(model_ckpt_path, images_dir, labels_dir, device='cuda'):
    # Load model (update args as needed to match your training)
    model = IRISModel(in_channels=1, base_channels=32, embed_dim=512, num_tokens=10, num_classes=13)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model.eval()
    model.to(device)
    inference = IRISInference(model, device=device)

    pairs = get_bcv_image_label_pairs(images_dir, labels_dir)
    print(f"Found {len(pairs)} BCV cases.")

    for i, (img_path, lbl_path) in enumerate(pairs):
        image, label = load_bcv_case(img_path, lbl_path)
        # Add batch and channel dims: (1, 1, D, H, W)
        image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
        # Dummy reference: use the same image/label as reference (for demo)
        reference_image = image_t
        reference_mask = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).to(device)
        # Run inference
        result = inference.one_shot_inference(image_t, reference_image, reference_mask)
        pred = result['prediction'].cpu().numpy().astype(np.uint8)[0,0]
        # Save prediction as NIfTI
        pred_nii = nib.Nifti1Image(pred, affine=np.eye(4))
        out_path = os.path.join('bcv_predictions', os.path.basename(img_path))
        os.makedirs('bcv_predictions', exist_ok=True)
        nib.save(pred_nii, out_path)
        print(f"[{i+1}/{len(pairs)}] Saved prediction to {out_path}")

if __name__ == "__main__":
    # Update these paths as needed
    model_ckpt = "path_to_trained_amos_model.pth"
    images_dir = "src/data/bcv/imagesTr"
    labels_dir = "src/data/bcv/labelsTr"
    main(model_ckpt, images_dir, labels_dir, device='cuda')
