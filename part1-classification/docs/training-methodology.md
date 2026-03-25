# Training Methodology

## Two-Stage Training Strategy

### Stage 1: Architecture Exploration (Open-Source Data)

**Objective:** Identify which CNN families perform best for chest X-ray classification.

**Dataset:** Kaggle chest X-ray — 7,135 images, 4 classes (COVID-19, Normal, Pneumonia, Tuberculosis)

**Approach:** Train 26 architectures from scratch with fixed hyperparameters. No pretrained weights — this tests each architecture's raw learning capacity.

| Setting | Value |
|---------|-------|
| Epochs | 10 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Input Sizes | 224×224 and 299×299 |
| Normalization | mean=0.5, std=0.5 (all channels) |

**Why from scratch?** To observe pure architectural differences. Pretrained weights would mask whether an architecture genuinely fits the task.

**Why fixed hyperparameters?** Fair comparison. Tuning each model individually would conflate architecture quality with tuning effort.

### Stage 2: Clinical Validation (Ethics-Approved Data)

**Objective:** Build production models for the KTÜ Farabi Hospital deployment.

**Dataset:** 2,024 ethics-approved chest X-rays, physician-labeled

**Approach:** Transfer learning — ImageNet1K_V1 pretrained, backbone frozen, only classifier retrained.

**Two rounds:**

| Round | Epochs | LR | Augmentation |
|-------|--------|----|--------------|
| Without fine-tuning | 5 | 0.001 | Resize + Normalize only |
| With fine-tuning | 7 | 0.0001 | Full augmentation pipeline |

**Why freeze backbone?** With ~809 training images, retraining all layers causes catastrophic overfitting. The ImageNet features (edges, textures, shapes) transfer well to medical imaging — we only need to teach the classifier what constitutes "pathology" in our specific context.

## The Binary Decomposition Decision

Initial plan: 4-class classification (Pathology+Suitable, Pathology+NotSuitable, NoPathy+Suitable, NoPathy+NotSuitable).

**Why it failed:**
- Some classes had <100 samples
- High inter-class variance with limited data
- WeightedRandomSampler and other balancing techniques didn't help sufficiently
- Multi-class confusion matrices showed systematic misclassification between similar classes

**Solution:** Two independent binary classifiers:
1. Pathology model: Present vs Absent
2. Evaluability model: Suitable vs Not Suitable

This reduced each problem's complexity and allowed each model to specialize.

## Augmentation Pipeline (Fine-Tuning Phase)

```
Input Image (299×299)
    │
    ├── CenterCrop(299, 224)
    ├── RandomRotation(30°)
    ├── RandomResizedCrop(299, scale=[0.8,1.0], ratio=[0.5,1.333])
    ├── RandomHorizontalFlip()
    ├── Pad(60, 20, 60, 20, fill=black)
    ├── ToTensor()
    └── Normalize(mean=0.5, std=0.5)
```

**Rationale for each transform:**
- **CenterCrop:** Focus on the central chest region where pathology is most likely
- **RandomRotation(30°):** X-rays can be slightly rotated in practice
- **RandomResizedCrop:** Scale variation simulates different patient-to-detector distances
- **RandomHorizontalFlip:** Anatomically valid — left-right chest symmetry
- **Padding:** Prevents information loss at image borders during rotation/cropping

## Auxiliary Output Handling

GoogLeNet and InceptionV3 have auxiliary classifiers used during training to combat vanishing gradients.

**Stage 1 (open-source):** Auxiliary outputs included with 0.4 weight:
```
loss = criterion(main_output, labels) + 0.4 × criterion(aux_output, labels)
```

**Stage 2 (ethics-approved):** Auxiliary outputs **disabled** — only main output used. With frozen backbones and transfer learning, auxiliary classifiers are unnecessary and can introduce noise.

## Hardware

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9 7950X (16 cores / 32 threads) |
| GPU | NVIDIA GeForce RTX 4060 Ti (16GB VRAM) |
| RAM | 60 GiB |
| CUDA | 12.8 |

Training a single architecture for 7 epochs on the ethics-approved dataset took approximately 2-5 minutes depending on model size.
