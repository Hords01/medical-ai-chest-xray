"""
CAM Pipeline for Chest X-Ray Explainability
=============================================
Illustrative implementation of the gradient-free XAI techniques used
in the clinical diagnostic support system.

Key insight: With frozen backbone layers, gradient-based methods
(GradCAM) can be misleading. We prioritize activation-based and
perturbation-based methods instead.
"""

import torch
import numpy as np
from typing import List, Dict, Optional

# In the actual system, pytorch-grad-cam library was used.
# pip install grad-cam
# from pytorch_grad_cam import (
#     EigenCAM, EigenGradCAM, ScoreCAM, AblationCAM
# )
# from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# CAM TECHNIQUE SELECTION RATIONALE
# ============================================================

CAM_TECHNIQUES = {
    "EigenCAM": {
        "type": "Activation-based (PCA)",
        "gradient_free": True,
        "description": "Uses principal component of feature map activations. "
                       "No gradient computation needed — ideal for frozen backbones.",
        "speed": "Fast",
        "stability": "High",
    },
    "EigenGradCAM": {
        "type": "Hybrid (PCA + Gradient)",
        "gradient_free": False,
        "description": "Combines eigendecomposition of activations with gradient "
                       "weighting. Provides richer signal than pure EigenCAM.",
        "speed": "Medium",
        "stability": "Medium-High",
    },
    "ScoreCAM": {
        "type": "Perturbation-based",
        "gradient_free": True,
        "description": "Uses forward-pass confidence scores as channel weights. "
                       "Completely gradient-free — gold standard for frozen models.",
        "speed": "Slow (multiple forward passes)",
        "stability": "High",
    },
    "AblationCAM": {
        "type": "Perturbation-based",
        "gradient_free": True,
        "description": "Systematically removes (ablates) feature channels and "
                       "measures impact on prediction. Very interpretable.",
        "speed": "Slow",
        "stability": "High",
    },
    "KPCA-CAM": {
        "type": "Kernel PCA on activations",
        "gradient_free": True,
        "description": "Non-linear activation decomposition using kernel PCA. "
                       "Captures complex, non-linear feature relationships that "
                       "standard PCA misses.",
        "speed": "Medium",
        "stability": "Medium",
    },
}


def compute_ensemble_cam(individual_cams: List[np.ndarray]) -> np.ndarray:
    """
    Ensemble Average CAM: Average all CAM heatmaps for stability.

    Why: Individual CAM techniques can be noisy and highlight
    different regions. Averaging produces a consensus explanation
    that is more robust and clinically trustworthy.

    Args:
        individual_cams: List of [H, W] heatmap arrays, each normalized to [0, 1]

    Returns:
        Averaged and re-normalized heatmap [H, W]
    """
    stacked = np.stack(individual_cams, axis=0)
    ensemble = np.mean(stacked, axis=0)

    # Re-normalize to [0, 1]
    ensemble = (ensemble - ensemble.min()) / (ensemble.max() - ensemble.min() + 1e-8)

    return ensemble


# ============================================================
# LIME GRID OVERLAY (3×4)
# ============================================================

def compute_lime_grid(model, image_tensor, device,
                       grid_rows=3, grid_cols=4,
                       num_perturbations=100) -> np.ndarray:
    """
    LIME-inspired grid analysis.

    Divides the image into a 3×4 grid (12 regions) and measures
    each region's contribution to the classification decision.

    Why 3×4?
    - Maps roughly to anatomical chest regions
    - 12 segments provide sufficient spatial resolution
    - Not so fine-grained that explanations become noisy

    Args:
        model: Classification model
        image_tensor: [1, C, H, W] input
        device: torch device
        grid_rows: Number of vertical divisions
        grid_cols: Number of horizontal divisions
        num_perturbations: Samples per region for LIME estimation

    Returns:
        [grid_rows, grid_cols] importance matrix
    """
    model.eval()
    _, C, H, W = image_tensor.shape
    cell_h = H // grid_rows
    cell_w = W // grid_cols

    # Baseline prediction
    with torch.no_grad():
        baseline_output = model(image_tensor.to(device))
        baseline_prob = torch.softmax(baseline_output, dim=1)
        predicted_class = baseline_prob.argmax(dim=1).item()
        baseline_confidence = baseline_prob[0, predicted_class].item()

    importance_grid = np.zeros((grid_rows, grid_cols))

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Occlude this region and measure impact
            occluded = image_tensor.clone()
            r_start = row * cell_h
            r_end = (row + 1) * cell_h if row < grid_rows - 1 else H
            c_start = col * cell_w
            c_end = (col + 1) * cell_w if col < grid_cols - 1 else W

            # Replace region with gray (mean value)
            occluded[:, :, r_start:r_end, c_start:c_end] = 0.0

            with torch.no_grad():
                occluded_output = model(occluded.to(device))
                occluded_prob = torch.softmax(occluded_output, dim=1)
                occluded_confidence = occluded_prob[0, predicted_class].item()

            # Importance = drop in confidence when region is removed
            importance_grid[row, col] = baseline_confidence - occluded_confidence

    # Normalize to [0, 1]
    if importance_grid.max() > importance_grid.min():
        importance_grid = (importance_grid - importance_grid.min()) / \
                          (importance_grid.max() - importance_grid.min())

    return importance_grid


# ============================================================
# FULL XAI PIPELINE
# ============================================================

def run_xai_pipeline(model, image_tensor, target_layer, device) -> Dict:
    """
    Run the complete XAI pipeline on a single image.

    Returns a dictionary with all explanation types:
    - Individual CAM heatmaps
    - Ensemble average CAM
    - LIME grid importance
    - Confidence metrics

    In the actual Streamlit app, these were displayed across
    tabs (Tab 1: XAI, Tab 2: Advanced Analysis, Tab 3: Report).
    """
    results = {}

    # Note: In the actual implementation, pytorch-grad-cam was used:
    #
    # from pytorch_grad_cam import EigenCAM, ScoreCAM, AblationCAM
    #
    # cam_methods = {
    #     "EigenCAM": EigenCAM(model=model, target_layers=[target_layer]),
    #     "ScoreCAM": ScoreCAM(model=model, target_layers=[target_layer]),
    #     "AblationCAM": AblationCAM(model=model, target_layers=[target_layer]),
    # }
    #
    # for name, cam in cam_methods.items():
    #     heatmap = cam(input_tensor=image_tensor)
    #     results[name] = heatmap[0]

    # Ensemble CAM
    # individual_cams = [results[name] for name in cam_methods]
    # results["Ensemble_Average_CAM"] = compute_ensemble_cam(individual_cams)

    # LIME Grid
    results["LIME_Grid"] = compute_lime_grid(model, image_tensor, device)

    return results
