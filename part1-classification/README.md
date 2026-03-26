# Part 1: Model Training, Ensemble Voting & Streamlit Deployment

## Overview

This phase covers the core machine learning pipeline: training 26 CNN architectures across two datasets, selecting the best performers, combining them via ensemble voting, and deploying the final system through Streamlit.

## The Problem

Chest X-ray interpretation requires expert radiologists, but in high-population cities the workload creates bottlenecks. In Turkey, consultation times can drop to 2-5 minutes per patient versus the WHO-recommended 20 minutes. An AI-assisted screening tool can serve as a second opinion, catching what fatigue might miss.

## Approach

### Phase 1a: Open-Source Dataset (Architecture Exploration)

**Dataset:** Kaggle chest X-ray dataset — 7,135 JPEG images across 4 classes (COVID-19, Normal, Pneumonia, Tuberculosis)

**Goal:** Identify which CNN families perform best for chest X-ray classification under standardized conditions.

**Method:**
- 26 architectures trained from scratch (no pretrained weights)
- All hyperparameters fixed: Adam optimizer, lr=0.001, CrossEntropyLoss, 10 epochs, batch_size=32
- Two input resolutions tested: 224×224 and 299×299
- Normalization: mean=0.5, std=0.5 across all 3 channels

**Key Finding:** Architecture design matters more than depth. MobileNetV3_small (1.5M params) achieved 82.49% accuracy, comparable to ResNeXt101_64x4d (81.4M params) at 83.14%.

### Phase 1b: Ethics-Approved Dataset (Clinical Validation)

**Dataset:** KTÜ Farabi Hospital — 2,024 PNG images, ethics board approved, physician-labeled

**Labels:**
| Model | Class 0 | Class 1 | Train | Test |
|-------|---------|---------|-------|------|
| Pathology | Pathology Absent | Pathology Present | 809 | 203 |
| Evaluability | Not Suitable | Suitable for Evaluation | 809 | 203 |

**Why Binary Instead of Multi-Class?**
Initial attempts at 4-class classification (combining both labels) failed due to insufficient samples per class and high inter-class imbalance. Decomposing into two binary problems dramatically improved results.

**Training Strategy:**
- ImageNet1K_V1 pretrained weights
- All layers frozen except final classification layer
- Only 8 architectures tested (top performers from Phase 1a)
- Two rounds: without fine-tuning (5 epochs, lr=0.001) and with fine-tuning (7 epochs, lr=0.0001 + augmentation)

### Phase 1c: Ensemble Voting

The top 3 models from fine-tuned training were combined:

| Model | Individual Accuracy | Individual NPV |
|-------|-------------------|----------------|
| DenseNet201 | 76.85% | **90.48%** |
| EfficientNet | 78.82% | 75.23% |
| ResNet101 | **79.31%** | 75.00% |

**Four voting strategies tested:**

| Strategy | How It Works |
|----------|--------------|
| Majority Voting | Each model gets one vote, majority wins |
| Soft Voting | Average predicted probabilities, highest wins |
| Weighted Soft Voting | Weighted average by model performance |
| Thresholded Soft Voting | Only predictions above a confidence threshold count |

**Results:**

| Method | Accuracy | NPV | F1 Score |
|--------|----------|-----|----------|
| Majority Voting | 84.03% | 82.28% | 0.8296 |
| **Soft Voting** | **86.11%** | **87.50%** | **0.8592** |
| **Weighted Soft Voting** | **86.11%** | **87.50%** | **0.8592** |
| Thresholded Soft Voting | 81.25% | 75.82% | 0.7805 |

Soft voting and weighted soft voting achieved identical best results — the probability-averaging mechanism effectively balanced the models' complementary strengths.

## Streamlit Deployment

The final system runs as a Streamlit web application:

```
User uploads chest X-ray
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│ Evaluability    │     │ Pathology        │
│ Model           │     │ Model (Ensemble) │
│                 │     │                  │
│ "Suitable" /    │     │ "Pathology       │
│ "Not Suitable"  │     │  Present" /      │
│                 │     │ "Absent"         │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│           Results Dashboard             │
│  • Classification results               │
│  • Sidebar: running statistics          │
│  • Session history                      │
└─────────────────────────────────────────┘
```

### Sidebar Statistics

The app maintains session-level statistics:
- Total images analyzed
- Count: Suitable / Not Suitable for evaluation
- Count: Pathology Present / Absent

## Key Takeaways

1. **Depth ≠ Performance** — DenseNet121 consistently outperformed DenseNet161/169 on NPV with fewer parameters
2. **NPV is the critical metric** — In clinical screening, missing a sick patient (false negative) is worse than a false alarm
3. **Ensemble > Individual** — Soft voting improved accuracy by ~7% and NPV by ~12% over the average individual model
4. **Binary decomposition works** — Splitting a hard 4-class problem into two focused binary problems was the key architectural decision
5. **Resolution matters** — GoogLeNet jumped from 82.88% to 87.55% accuracy simply by changing from 224×224 to 299×299

## Documentation

- [Architecture Comparison (26 models)](docs/architecture-comparison.md)
- [Training Methodology](docs/training-methodology.md)
- [Ensemble Voting Deep-Dive](docs/ensemble-voting.md)
- [Complete Results Tables](docs/results-tables.md)

## Example Code

- [Training Pipeline Pattern](examples/training_pipeline.py)
- [Ensemble Voting Pattern](examples/ensemble_voting.py)
- [Streamlit App Pattern](examples/streamlit_app.py)
