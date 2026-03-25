# XAI Techniques Deep-Dive

## Overview of Techniques

Six Class Activation Mapping variants plus LIME-based regional analysis were integrated into the Streamlit clinical interface.

## CAM Variants

### EigenCAM
**Type:** Activation-based (PCA decomposition)

Computes the first principal component of the feature map activations at a target layer. The resulting heatmap shows which spatial regions have the highest-variance activation patterns — regions the network "pays most attention to."

**Advantage:** Completely gradient-free. Works identically whether layers are frozen or trainable.

### EigenGradCAM
**Type:** Hybrid (PCA + gradient weighting)

Similar to EigenCAM but incorporates gradient information to weight the principal components. The gradient from the output class score provides task-specific direction to the activation decomposition.

**Advantage:** More task-specific than pure EigenCAM while still being robust to frozen-layer gradient issues through the eigendecomposition component.

### ScoreCAM
**Type:** Perturbation-based (forward-pass scoring)

For each activation channel, creates a binary mask, upsamples it to input size, applies it to the input image, and measures the resulting confidence score via a forward pass. Channels that produce higher confidence when applied as masks receive higher importance.

**Advantage:** Zero gradient computation. The gold standard for gradient-free explainability.

**Disadvantage:** Requires N forward passes (one per channel) — significantly slower than gradient-based methods.

### AblationCAM
**Type:** Perturbation-based (ablation study)

Systematically zeroes out each activation channel and measures the drop in predicted confidence. Channels whose removal causes the largest confidence drop are most important.

**Advantage:** Highly interpretable — directly answers "how much does this feature matter?"

### KPCA-CAM
**Type:** Kernel PCA on activations

Applies kernel PCA (non-linear dimensionality reduction) to the activation maps. Captures complex, non-linear relationships between activation patterns that standard PCA misses.

**Advantage:** Can reveal subtle pathological patterns that linear methods overlook.

### Ensemble Average CAM
**Type:** Meta-technique (consensus)

Averages the normalized heatmaps from all five CAM techniques above. Provides the most stable and reliable explanation by filtering out technique-specific biases.

## LIME Grid Overlay (3×4)

Unlike CAM methods that operate on model internals, LIME treats the model as a black box and perturbs the input.

**Process:**
1. Divide the 299×299 image into a 3×4 grid (12 regions)
2. For each region, create perturbations (occlude the region)
3. Measure the prediction change when each region is hidden
4. Regions that cause the largest prediction shift are most important

**Why 3×4?** This grid roughly corresponds to the anatomical regions visible in a standard PA (posteroanterior) chest X-ray — upper/mid/lower zones × left/right + central.

## Regional Analysis

The 3×4 grid regional analysis provides a structured spatial breakdown of prediction contributions. Each cell in the grid receives an importance score, enabling radiologists to see at a glance which anatomical regions drove the AI's decision.

This was displayed as a color-coded overlay on the original X-ray image in the Streamlit interface.
