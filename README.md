# Medical Image Classification & AI-Assisted Diagnostic System

> **Chest X-Ray Pathology Detection using Deep Learning Ensemble Methods with Explainable AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)]()

## Overview

This repository documents the full development lifecycle of a **clinical chest X-ray diagnostic support system**, originally developed as a Bachelor's thesis at Karadeniz Technical University (KTÜ), Department of Statistics and Computer Science.

The system was **deployed and actively used at KTÜ Farabi Hospital** under ethics board approval, providing radiologists with AI-assisted pathology detection and explainable decision support.

### What This Repository Contains

This is a **project documentation and methodology repository** — not a code dump. It serves as a comprehensive record of the design decisions, experimental results, architectural patterns, and lessons learned throughout the project. Representative code examples are included to illustrate key implementations.

> **Note:** The original codebase and trained model weights remain with KTÜ. No patient data, ethically-approved medical images, or model weights trained on clinical data are included in this repository.

---

## Project Architecture

The project evolved through three distinct phases:

```
Phase 1: Classification          Phase 2: XAI Integration       Phase 3: ML Framework
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ • 26 CNN architectures   │     │ • 6 CAM techniques        │     │ • YAML config system      │
│   trained & compared     │     │ • LIME grid overlay       │     │ • 5 checkpoint strategies │
│ • Fine-tuning pipeline   │     │ • Confidence analysis     │     │ • DICOM/CLAHE support     │
│ • Ensemble voting        │     │ • Uncertainty metrics     │     │ • Auto model selection    │
│ • Streamlit deployment   │     │ • Streamlit XAI tabs      │     │ • Multi-format export     │
│                          │     │                           │     │                           │
│ Result: 86.11% accuracy  │     │ Result: Interpretable     │     │ Result: Production-ready  │
│ 87.50% NPV               │     │ clinical decision support │     │ training pipeline         │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## Key Results

| Metric | Best Single Model | Ensemble (Soft Voting) |
|--------|-------------------|------------------------|
| **Accuracy** | 79.31% (ResNet101) | **86.11%** |
| **NPV** | 90.48% (DenseNet201) | **87.50%** |
| **F1 Score** | 0.7926 (ResNet101) | **0.8592** |
| **Sensitivity** | 79.31% (ResNet101) | **87.14%** |
| **Precision** | 80.25% (DenseNet201) | **84.72%** |

> **Why NPV matters:** In clinical screening, a high Negative Predictive Value means that when the model says "no pathology," you can trust it. A missed diagnosis (false negative) is far more dangerous than a false alarm.

## Repository Structure

```
medical-ai-chest-xray/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── DISCLAIMER.md                      # Clinical & ethical disclaimers
├── ETHICS.md                          # Ethics statement & data handling
│
├── part1-classification/              # Phase 1: Model Training & Deployment
│   ├── README.md                      # Detailed phase overview
│   ├── docs/
│   │   ├── architecture-comparison.md # 26 architectures analyzed
│   │   ├── training-methodology.md    # Training pipeline details
│   │   ├── ensemble-voting.md         # 4 voting strategies compared
│   │   └── results-tables.md          # Complete experimental results
│   └── examples/
│       ├── training_pipeline.py       # Representative training code
│       ├── ensemble_voting.py         # Ensemble implementation pattern
│       └── streamlit_app.py           # Streamlit deployment pattern
│
├── part2-xai-integration/             # Phase 2: Explainable AI
│   ├── README.md                      # XAI techniques & motivation
│   ├── docs/
│   │   ├── xai-techniques.md          # CAM variants, LIME, analysis
│   │   ├── confidence-analysis.md     # Uncertainty & entropy metrics
│   │   └── gradient-free-rationale.md # Why gradient-free XAI was chosen
│   └── examples/
│       ├── cam_pipeline.py            # CAM technique patterns
│       ├── confidence_metrics.py      # Confidence/uncertainty calculation
│       └── streamlit_xai_app.py       # XAI-enhanced Streamlit pattern
│
├── part3-training-framework/          # Phase 3: ML Training Framework
│   ├── README.md                      # Framework overview & philosophy
│   ├── docs/
│   │   ├── config-system.md           # YAML configuration deep-dive
│   │   ├── checkpoint-management.md   # 5 checkpoint strategies
│   │   ├── medical-imaging.md         # DICOM windowing, CLAHE
│   │   ├── metrics-system.md          # NPV, AUC-ROC, custom metrics
│   │   └── test-only-mode.md          # Evaluation without retraining
│   └── examples/
│       ├── config_binary.yaml         # Binary classification config
│       ├── config_multiclass.yaml     # Multi-class config
│       ├── config_medical.yaml        # Medical imaging config
│       └── framework_architecture.py  # Framework class structure
│
└── assets/                            # Diagrams and figures (no patient data)
    └── .gitkeep
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| Deep Learning Framework | PyTorch 2.0+ |
| CNN Architectures | DenseNet, ResNet, EfficientNet, MobileNet, ShuffleNet, VGG, AlexNet, GoogLeNet, InceptionV3, ResNeXt |
| Explainable AI | pytorch-grad-cam, LIME, custom CAM implementations |
| Application | Streamlit |
| Training Data | Kaggle open-source chest X-ray dataset (7,135 images) + KTÜ Farabi Hospital ethics-approved dataset (2,024 images) |
| Hardware | AMD Ryzen 9 7950X, NVIDIA RTX 4060 Ti 16GB, 60GB RAM |
| Metrics | Accuracy, Precision, Recall, F1, NPV, AUC-ROC, Confusion Matrix |

## The Two-Model Architecture

Rather than a single multi-class classifier, the system uses **two specialized binary classifiers**:

```
                    ┌─────────────────────┐
   Chest X-Ray ───▶│  Model 1: Pathology  │───▶ Pathology Present / Absent
     Input          │  (DenseNet201 +      │
                    │   EfficientNet +     │
                    │   ResNet101 Ensemble)│
                    └─────────────────────┘
                    ┌─────────────────────┐
                ───▶│  Model 2: Quality    │───▶ Suitable / Not Suitable
                    │  (Evaluability)      │     for Evaluation
                    └─────────────────────┘
```

**Why two models instead of four classes?**
With limited clinical data (~2,024 images), multi-class classification showed high variance and poor performance. Decomposing into two binary problems reduced complexity and significantly improved results.

## Development Timeline

| Phase | Period | Scope |
|-------|--------|-------|
| **Phase 1** | Thesis period | 26 architectures trained, ensemble voting, Streamlit deployment, thesis defense |
| **Phase 2** | Post-thesis (self-initiated) | XAI integration, confidence metrics, enhanced clinical interface |
| **Phase 3** | Pre-graduation | Reusable ML training framework for future research |

## Acknowledgments

This work was conducted at the **Department of Statistics and Computer Science, Faculty of Science, Karadeniz Technical University**.

- **Thesis Advisor:** Asst. Prof. Üyesi Tolga BERBER
- **Thesis Committee:** Asst. Prof. Üyesi Tolga BERBER, Asst. Prof. Üyesi Uğur ŞEVİK, Asst. Prof. Üyesi Halil İbrahim ŞAHİN
- **Clinical Data:** KTÜ Farabi Hospital (ethics board approved, data remains with the institution)

## Explore Each Phase

| Phase | What You'll Find |
|-------|------------------|
| [**Part 1: Classification**](part1-classification/) | How 26 architectures were compared, why ensemble voting works, complete result tables |
| [**Part 2: XAI Integration**](part2-xai-integration/) | Why gradient-free XAI was chosen, how confidence metrics work, the clinical interface |
| [**Part 3: Training Framework**](part3-training-framework/) | YAML-driven training, checkpoint strategies, medical imaging preprocessing |

---

## Author

**Emirkan Beyaz**
- BSc in Statistics and Computer Science, Karadeniz Technical University
- Focus: Computer Vision, Medical AI, Explainable AI

## License

This documentation and example code is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> **Important:** This license covers only the documentation and representative code examples in this repository. Clinical data, trained model weights, and the deployed system remain the property of KTÜ and are not included here.
