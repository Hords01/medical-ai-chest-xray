# Part 2: Explainable AI (XAI) Integration

## Overview

This phase was **self-initiated** after the thesis submission. During an extended semester, I integrated explainable AI techniques into the Streamlit-based clinical interface, transforming the system from a black-box classifier into an interpretable diagnostic support tool.

## Motivation: Why XAI in Medical AI?

A model that says "Pathology Present" with 86% accuracy is useful. A model that says "Pathology Present — here's why, and here's how confident I am" is **trustworthy**. In clinical settings, physicians need to understand *why* the AI reached its conclusion before incorporating it into their diagnostic process.

## The Technical Challenge: Frozen Layers

A critical constraint shaped our XAI approach:

```
┌──────────────────────────────────────────────────┐
│           Model Architecture                      │
│                                                    │
│  ┌────────────────────────┐  FROZEN               │
│  │  Pretrained Backbone   │  (ImageNet weights)    │
│  │  - Conv layers         │  → Gradients may be    │
│  │  - Feature extractors  │    MISLEADING here     │
│  └────────────┬───────────┘                        │
│               │                                    │
│  ┌────────────▼───────────┐  TRAINABLE             │
│  │  Classification Head   │  (Retrained on our     │
│  │  - Final FC layer      │   clinical data)       │
│  └────────────────────────┘                        │
└──────────────────────────────────────────────────┘
```

Because the backbone layers were **frozen during training** (only the final classifier was retrained), gradient-based XAI methods like standard GradCAM could produce misleading explanations — the gradients in early layers reflect ImageNet features, not our clinical task. This led us to prioritize **gradient-free and activation-based** XAI techniques.

## XAI Techniques Used

### CAM-Based Methods

| Technique | Type | Why Chosen |
|-----------|------|------------|
| **EigenCAM** | Activation-based (PCA) | Uses principal component of activations — no gradients needed |
| **EigenGradCAM** | Hybrid | Combines eigendecomposition with gradient weighting |
| **ScoreCAM** | Perturbation-based | Uses forward-pass scores as weights — fully gradient-free |
| **AblationCAM** | Perturbation-based | Systematically removes features to measure importance |
| **KPCA-CAM** | Kernel PCA on activations | Non-linear activation decomposition — captures complex patterns |
| **Ensemble Average CAM** | Meta-technique | Averages all CAM outputs for stability and reduced variance |

### LIME-Based Analysis

**LIME Grid Overlay (3×4):**
- Divides the image into a 3×4 grid (12 regions)
- Measures each region's contribution to the classification decision
- Uses superpixel-based local explanations
- Provides intuitive spatial contribution mapping

### Regional Analysis

**3×4 Grid Regional Analysis:**
- Systematic analysis of how different anatomical regions contribute to the prediction
- Identifies which areas of the chest X-ray drove the classification

## Confidence & Uncertainty Metrics

Beyond visual explanations, the system provides quantitative confidence measures:

### Metrics Computed

| Metric | Formula | Clinical Meaning |
|--------|---------|-----------------|
| **Confidence** | max(softmax output) | How sure is the model about its top prediction? |
| **Uncertainty** | 1 - confidence | How unsure is the model? |
| **Entropy** | -Σ p·log(p) | Information-theoretic measure of prediction uncertainty |
| **Class 0 Probability** | softmax[0] | Probability of "no pathology" |
| **Class 1 Probability** | softmax[1] | Probability of "pathology present" |

### Confidence Interpretation System

The system translates raw confidence scores into clinician-friendly categories:

| Confidence Range | Category (TR) | Category (EN) | Guidance |
|------------------|---------------|---------------|----------|
| ≥ 0.95 | Çok Yüksek Güven | Very High Confidence | Model is very certain |
| ≥ 0.85 | Yüksek Güven | High Confidence | Model is quite certain |
| ≥ 0.70 | Orta Güven | Moderate Confidence | Model is moderately certain |
| ≥ 0.55 | Düşük Güven | Low Confidence | Model is uncertain — second opinion recommended |
| < 0.55 | Çok Düşük Güven | Very Low Confidence | Model cannot decide — expert review required |

## Streamlit Application Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     MAIN PAGE                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Upload Chest X-Ray Image                             │   │
│  │  [Classification Results: Pathology + Evaluability]   │   │
│  │  [Original Image Display]                              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  DETAILED ANALYSIS (TABS)                              │   │
│  │                                                        │   │
│  │  ┌─────────┐  ┌───────────────┐  ┌─────────┐          │   │
│  │  │ TAB 1:  │  │ TAB 2:        │  │ TAB 3:  │          │   │
│  │  │ XAI     │  │ ADVANCED      │  │ REPORT  │          │   │
│  │  │         │  │ ANALYSIS      │  │         │          │   │
│  │  │•EigenCAM│  │•Confidence    │  │•Summary │          │   │
│  │  │•ScoreCAM│  │•Uncertainty   │  │•Export  │          │   │
│  │  │•Ablation│  │•Entropy       │  │         │          │   │
│  │  │•KPCA-CAM│  │•Class Probs   │  │         │          │   │
│  │  │•Ensemble│  │•Interpretation│  │         │          │   │
│  │  │•LIME    │  │               │  │         │          │   │
│  │  │•Regional│  │               │  │         │          │   │
│  │  └─────────┘  └───────────────┘  └─────────┘          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────┐                                    │
│  │  SIDEBAR              │                                    │
│  │  📊 Prediction Summary│                                    │
│  │  • Total analyzed      │                                    │
│  │  • Suitable count      │                                    │
│  │  • Not suitable count  │                                    │
│  │  • Pathology present   │                                    │
│  │  • Pathology absent    │                                    │
│  └──────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Gradient-free priority:** Frozen backbone layers make gradient-based explanations unreliable for early layers → chose ScoreCAM, AblationCAM, EigenCAM

2. **Ensemble CAM for stability:** Individual CAM techniques can be noisy. Averaging all 5 CAM outputs produces more stable, consistent explanations

3. **12-region LIME grid:** Medical image interpretation is inherently spatial. A 3×4 grid maps roughly to anatomical quadrants of the chest

4. **Dual confidence display:** Both the pathology model and evaluability model display their confidence metrics separately, allowing physicians to assess trust in each classification independently

5. **Turkish-language interface:** The clinical deployment at KTÜ Farabi Hospital required a Turkish interface with appropriate medical terminology

## Documentation

- [XAI Techniques Deep-Dive](docs/xai-techniques.md)
- [Confidence Analysis System](docs/confidence-analysis.md)
- [Why Gradient-Free XAI](docs/gradient-free-rationale.md)

## Example Code

- [CAM Pipeline Pattern](examples/cam_pipeline.py)
- [Confidence Metrics Calculation](examples/confidence_metrics.py)
- [XAI-Enhanced Streamlit Pattern](examples/streamlit_xai_app.py)
