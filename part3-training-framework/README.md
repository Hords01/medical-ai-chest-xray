# Part 3: ML Training Framework for Medical Image Classification

## Overview

Before graduating, I built a **reusable, production-ready training framework** so that future researchers at KTÜ could train new models as clinical data grows — without rewriting infrastructure code each time.

The framework's philosophy: **configure, don't code.** A single YAML file controls the entire pipeline — from data loading to checkpoint management to metric reporting.

## The Problem It Solves

Every new medical imaging project at the lab faced the same overhead:

```
❌ Before: Every new project
├── Write data loading code from scratch
├── Set up logging manually
├── Implement metrics tracking
├── Build checkpoint management
├── Handle DICOM preprocessing
├── Create confusion matrices
├── Export results to different formats
├── Write experiment documentation
└── Debug the same issues again
```

```
✅ After: With the framework
├── Copy a YAML template
├── Set 3 values: data path, num_classes, validation path
├── Run: python train.py --config my_config.yaml
└── Get: trained model + experiment report + all metrics
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    config.yaml (Single Source of Truth)          │
│  ┌───────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐│
│  │   Data     │ │   Training   │ │  Checkpoint  │ │  Medical  ││
│  │  Settings  │ │  Settings    │ │   Strategy   │ │  Imaging  ││
│  └─────┬─────┘ └──────┬───────┘ └──────┬───────┘ └─────┬─────┘│
└────────┼───────────────┼────────────────┼───────────────┼──────┘
         │               │                │               │
         ▼               ▼                ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  DataLoader  │ │   Trainer    │ │  Checkpoint  │ │  Preprocess  │
│  • ImageFolder│ │  • Train loop│ │  Manager     │ │  Pipeline    │
│  • CSV format │ │  • Validation│ │  • 5 strategies│ │ • Windowing │
│  • Auto-detect│ │  • Metrics   │ │  • Cleanup   │ │  • CLAHE    │
│  • Sampling  │ │  • TensorBoard│ │  • Best model│ │  • DICOM    │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
         │               │                │               │
         └───────────────┴────────────────┴───────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │      Output Package       │
                    │  • best_model.pth         │
                    │  • metrics.json/csv/xlsx  │
                    │  • confusion_matrices/    │
                    │  • tensorboard_logs/      │
                    │  • experiment_README.md   │
                    └──────────────────────────┘
```

## Key Features

### 1. YAML-Driven Configuration (40+ Toggle Options)

Everything is controlled from a single YAML file. No code changes needed between experiments.

```yaml
# Example: Binary chest X-ray classification
data:
  train_data_path: "./data/pathology/train"
  test_data_path: "./data/pathology/test"
  val_data_path: null  # Framework adapts: uses train metrics for model selection
  num_classes: 2
  image_size: 299
  format: "imagefolder"  # or "csv"

training:
  epochs: 50
  learning_rate: 0.0001
  optimizer: "adam"
  criterion: "cross_entropy"
  batch_size: 32

preprocessing:
  normalize: true
  random_horizontal_flip: true
  random_rotation: 30
  random_resized_crop: true
  color_jitter: false  # Disable what you don't need
```

### 2. Checkpoint Management (5 Strategies)

**The problem:** Early versions saved a checkpoint every epoch AND every best model update. A 100-epoch run produced 200+ files, consuming gigabytes.

**The solution:** 5 configurable strategies:

| Strategy | Files Created | Use Case |
|----------|--------------|----------|
| `none` | 0 | Quick experiments, don't need to save |
| `best_only` | 1 (best_model.pth) | Production training — just give me the best |
| `best_and_latest` | 2 (best + latest) | **Default** — resume training + best model |
| `every_n_epochs` | N/epochs files | Periodic snapshots for analysis |
| `all` | epochs × 2 files | Full history (research/debugging) |

```yaml
checkpoint:
  strategy: "best_and_latest"  # Default: 100 epochs → only 2 files
  save_only_simple_best: true
  save_only_simple_latest: true
  cleanup_old_best: true
```

### 3. Medical Imaging Preprocessing

**DICOM Windowing:**
Medical DICOM images have Hounsfield Unit values ranging from -1000 to +3000. Different tissue types need different "windows":

| Tissue | Window Center | Window Width | Purpose |
|--------|--------------|--------------|---------|
| Lung | -600 | 1500 | Airway and parenchyma visibility |
| Soft Tissue | 40 | 400 | Organ boundaries |
| Bone | 400 | 1800 | Skeletal structures |

```yaml
medical_imaging:
  enabled: true
  windowing:
    enabled: true
    window_center: -600
    window_width: 1500
  clahe:
    enabled: true
    clip_limit: 2.0
    tile_grid_size: [8, 8]
```

**CLAHE (Contrast Limited Adaptive Histogram Equalization):**
Low-contrast X-ray images are enhanced locally, making subtle features more visible to the model without introducing global artifacts.

### 4. Smart Model Selection

The framework doesn't just pick the model with the highest accuracy. It supports **weighted multi-metric selection:**

```yaml
model_selection:
  strategy: "weighted"  # or "single_metric"
  metrics:
    accuracy: 0.40
    f1_score: 0.30
    npv: 0.20
    auc_roc: 0.10
```

This was critical for clinical applications where NPV matters more than raw accuracy.

### 5. Test-Only Mode

Evaluate a previously trained model without retraining:

```yaml
data:
  train_data_path: null  # Triggers test-only mode
  test_data_path: "./data/test"

test_only:
  checkpoint_path: "./models/best_model.pth"
  strict_loading: false  # Partial loading if architecture differs slightly
```

The `strict=False` option handles cases where model architecture was updated — it loads matching keys and reports which ones are missing/unexpected.

### 6. Multi-Format Metrics Export

Every training run automatically exports:

| Format | Content | Use Case |
|--------|---------|----------|
| JSON | All metrics, all epochs | Programmatic access |
| CSV | Epoch-by-epoch metrics | Spreadsheet analysis |
| Excel | Formatted tables | Reports and presentations |
| TensorBoard | Real-time graphs | Live monitoring |
| Confusion Matrix | Per-epoch PNG images | Visual error analysis |
| README | Experiment summary | Reproducibility documentation |

### 7. Flexible Data Loading

The framework auto-detects data format:

**ImageFolder structure:**
```
data/
├── train/
│   ├── class_0/
│   │   ├── img001.png
│   │   └── ...
│   └── class_1/
│       ├── img001.png
│       └── ...
└── test/
    ├── class_0/
    └── class_1/
```

**CSV format:**
```csv
image_path,label
/data/images/img001.png,0
/data/images/img002.png,1
```

### 8. Requirements by Profile

```
requirements/
├── minimal.txt    # Core: torch, torchvision, numpy, pillow
├── full.txt       # All features: + tensorboard, openpyxl, matplotlib
├── dev.txt        # Development: + pytest, black, mypy
└── medical.txt    # Medical imaging: + pydicom, nibabel, SimpleITK
```

## Augmentation Options (11 Configurable)

All toggleable from YAML — enable only what your data needs:

| Augmentation | Config Key | Default |
|--------------|------------|---------|
| Random Horizontal Flip | `random_horizontal_flip` | true |
| Random Vertical Flip | `random_vertical_flip` | false |
| Random Rotation | `random_rotation` | 15° |
| Random Resized Crop | `random_resized_crop` | true |
| Color Jitter | `color_jitter` | false |
| Random Affine | `random_affine` | false |
| Gaussian Blur | `gaussian_blur` | false |
| Random Erasing | `random_erasing` | false |
| Center Crop | `center_crop` | false |
| Normalize | `normalize` | true |
| Padding | `padding` | false |

## Supported Metrics

| Metric | Binary | Multi-Class | Notes |
|--------|--------|-------------|-------|
| Accuracy | ✅ | ✅ | |
| Precision | ✅ | ✅ | Macro & weighted |
| Recall | ✅ | ✅ | Macro & weighted |
| F1 Score | ✅ | ✅ | Macro & weighted |
| NPV | ✅ | ✅ | Critical for medical applications |
| AUC-ROC | ✅ | ✅ | OvR and OvO strategies |
| Confusion Matrix | ✅ | ✅ | Per-epoch visualization |
| Per-Class Metrics | ✅ | ✅ | Optional detailed breakdown |

## Bug Fixes & Production Hardening

During development, several critical issues were identified and resolved:

1. **Checkpoint file explosion** — Solved with 5 configurable strategies
2. **YAML parsing failures** — Added missing config classes (RandomAffineConfig, MedicalImagingConfig, WindowingConfig, CLAHEConfig)
3. **UserWarning spam** — Suppressed deprecated `pretrained` parameter warnings
4. **Test-only validation warning** — Removed unnecessary warning when no validation set exists
5. **Checkpoint loading mismatches** — Added `strict=False` with detailed mismatch reporting

## Documentation

- [Configuration System Deep-Dive](docs/config-system.md)
- [Checkpoint Management Strategies](docs/checkpoint-management.md)
- [Medical Imaging Preprocessing](docs/medical-imaging.md)
- [Metrics System & NPV](docs/metrics-system.md)
- [Test-Only Mode](docs/test-only-mode.md)

## Example Configs

- [Binary Classification](examples/config_binary.yaml)
- [Multi-Class Classification](examples/config_multiclass.yaml)
- [Medical Imaging with DICOM](examples/config_medical.yaml)
- [Framework Class Structure](examples/framework_architecture.py)
