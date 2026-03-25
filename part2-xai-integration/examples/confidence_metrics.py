"""
Confidence & Uncertainty Metrics
=================================
Quantitative measures of model confidence for clinical decision support.

These metrics complement the visual XAI explanations (CAM, LIME) by
providing numerical answers to "how sure is the model?"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


def compute_confidence_metrics(model, image_tensor, device) -> Dict[str, float]:
    """
    Compute comprehensive confidence metrics for a single prediction.

    Args:
        model: Trained classification model
        image_tensor: Preprocessed image tensor [1, C, H, W]
        device: torch device

    Returns:
        Dictionary with all confidence metrics
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze()

    probs_np = probabilities.cpu().numpy()

    # Core metrics
    confidence = float(np.max(probs_np))
    uncertainty = 1.0 - confidence
    predicted_class = int(np.argmax(probs_np))

    # Shannon entropy: higher = more uncertain
    # For binary: max entropy = log(2) ≈ 0.693 when p = [0.5, 0.5]
    entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "entropy": entropy,
        "class_0_prob": float(probs_np[0]),
        "class_1_prob": float(probs_np[1]),
    }


def get_confidence_interpretation(confidence: float) -> Tuple[str, str]:
    """
    Translate raw confidence scores into clinician-friendly categories.

    Design rationale: Physicians don't think in probabilities.
    Categories like "High Confidence" and actionable guidance like
    "second opinion recommended" bridge the gap between model output
    and clinical workflow.

    Returns:
        Tuple of (category_name, explanation)
    """
    if confidence >= 0.95:
        return (
            "Çok Yüksek Güven",
            "Model bu tahminden çok emin"
        )
    elif confidence >= 0.85:
        return (
            "Yüksek Güven",
            "Model bu tahminden oldukça emin"
        )
    elif confidence >= 0.70:
        return (
            "Orta Güven",
            "Model bu tahminden orta derecede emin"
        )
    elif confidence >= 0.55:
        return (
            "Düşük Güven",
            "Model kararsız, ikinci görüş önerilir"
        )
    else:
        return (
            "Çok Düşük Güven",
            "Model karar veremiyor, uzman incelemesi gerekli"
        )


def get_uncertainty_analysis(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Provide a comprehensive uncertainty analysis for clinical display.

    Combines multiple signals:
    - Raw confidence
    - Entropy level
    - Class probability gap (how far apart the two classes are)
    """
    confidence = metrics["confidence"]
    entropy = metrics["entropy"]
    prob_gap = abs(metrics["class_1_prob"] - metrics["class_0_prob"])

    analysis = {}

    # Confidence interpretation
    category, explanation = get_confidence_interpretation(confidence)
    analysis["confidence_category"] = category
    analysis["confidence_explanation"] = explanation

    # Entropy interpretation
    # Binary classification max entropy ≈ 0.693
    if entropy < 0.2:
        analysis["entropy_level"] = "Düşük Entropi — Kesin karar"
    elif entropy < 0.5:
        analysis["entropy_level"] = "Orta Entropi — Belirli düzeyde belirsizlik"
    else:
        analysis["entropy_level"] = "Yüksek Entropi — Önemli belirsizlik"

    # Probability gap interpretation
    if prob_gap > 0.5:
        analysis["decision_clarity"] = "Net ayrım — sınıflar arası fark belirgin"
    elif prob_gap > 0.2:
        analysis["decision_clarity"] = "Orta ayrım — sınıflar kısmen yakın"
    else:
        analysis["decision_clarity"] = "Belirsiz ayrım — sınıf olasılıkları çok yakın"

    # Clinical recommendation
    if confidence >= 0.85 and entropy < 0.3:
        analysis["recommendation"] = "Sonuç güvenilir görünüyor"
    elif confidence >= 0.70:
        analysis["recommendation"] = "Sonuç kabul edilebilir, ancak dikkatli incelenmeli"
    else:
        analysis["recommendation"] = "İkinci görüş veya ek tetkik önerilir"

    return analysis


# ============================================================
# Dual-Model Confidence Display
# ============================================================

def compute_dual_model_analysis(pathology_model, evaluability_model,
                                 image_tensor, device) -> Dict:
    """
    Compute and combine confidence metrics from both models.

    In the deployed system, both models' results were displayed
    side-by-side so the physician could assess:
    1. Is this image suitable for evaluation? (quality gate)
    2. Does this image show pathology? (clinical finding)
    """
    path_metrics = compute_confidence_metrics(pathology_model, image_tensor, device)
    eval_metrics = compute_confidence_metrics(evaluability_model, image_tensor, device)

    path_analysis = get_uncertainty_analysis(path_metrics)
    eval_analysis = get_uncertainty_analysis(eval_metrics)

    # Class label mapping
    pathology_labels = {0: "Patoloji Yok", 1: "Patoloji Var"}
    evaluability_labels = {0: "Değerlendirilmeye Uygun Değil", 1: "Değerlendirilmeye Uygun"}

    return {
        "pathology": {
            "prediction": pathology_labels[path_metrics["predicted_class"]],
            "metrics": path_metrics,
            "analysis": path_analysis,
        },
        "evaluability": {
            "prediction": evaluability_labels[eval_metrics["predicted_class"]],
            "metrics": eval_metrics,
            "analysis": eval_analysis,
        }
    }
