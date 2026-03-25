# Why Gradient-Free XAI Was Chosen

## The Frozen Backbone Problem

In our transfer learning approach, all backbone layers were frozen (ImageNet weights) and only the final classification layer was retrained on clinical data. This creates a fundamental issue for gradient-based explainability methods.

### How Standard GradCAM Works

GradCAM computes the gradient of the predicted class score with respect to the feature maps of a chosen convolutional layer:

```
α_k = (1/Z) Σ_i Σ_j (∂y_c / ∂A_k_ij)      ← gradient weights
L_GradCAM = ReLU(Σ_k α_k · A_k)              ← weighted activation sum
```

The problem: **gradients flow through the frozen backbone, but those layers were trained on ImageNet (dogs, cats, cars) — not chest X-rays.** The gradient signal from the trainable classifier passes through layers that learned generic features, not task-specific ones.

### What Goes Wrong

```
Trainable Classifier (learns "pathology patterns")
        │
        │ ← gradients are meaningful here
        │
Frozen Layer N (generic ImageNet features)
        │ ← gradients may be MISLEADING here
        │    because these weights were optimized for
        │    ImageNet, not chest X-ray pathology
Frozen Layer 1
```

The early frozen layers may highlight features that were important for ImageNet classification but are irrelevant or misleading for pathology detection. A GradCAM heatmap might highlight textural patterns that correlate with ImageNet classes rather than genuine pathological indicators.

## Our Solution: Gradient-Free and Activation-Based Methods

### Methods That Don't Need Gradients

| Method | How It Works | Why It's Reliable With Frozen Layers |
|--------|-------------|--------------------------------------|
| **EigenCAM** | PCA of activation maps → principal component | Only looks at what the network activates on, not how gradients flow |
| **ScoreCAM** | Masks each channel, measures confidence change via forward passes | Zero gradient computation — purely observation-based |
| **AblationCAM** | Removes channels, observes impact | Same principle — tests "what happens if this feature is missing?" |
| **KPCA-CAM** | Kernel PCA on activations | Non-linear decomposition of activation patterns |

### The One Hybrid Method

**EigenGradCAM** uses gradients but weighs them with eigendecomposition. We included it because:
1. The gradient signal from the final trainable layer is still meaningful
2. Combining it with activation-based eigendecomposition reduces noise from frozen layer gradients
3. It provides a point of comparison between pure gradient-free and hybrid approaches

## Ensemble Average CAM: The Clinical-Safe Choice

Individual CAM techniques each have biases. Our solution was to average all five CAM outputs:

```
Ensemble_CAM = (1/5) × (EigenCAM + EigenGradCAM + ScoreCAM + AblationCAM + KPCA_CAM)
```

This produces **consensus explanations**: regions that multiple techniques agree on are highlighted strongly, while technique-specific artifacts cancel out. For clinical use, this stability is more important than any single technique's theoretical optimality.

## Validation

While formal XAI evaluation (e.g., insertion/deletion metrics, pointing game) was beyond this thesis's scope, informal validation with supervising physicians confirmed that the Ensemble Average CAM heatmaps consistently highlighted anatomically relevant regions (lung fields, cardiac silhouette borders, costophrenic angles) rather than artifacts or irrelevant structures.
