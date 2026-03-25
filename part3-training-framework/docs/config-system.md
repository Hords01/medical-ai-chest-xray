# YAML Configuration System

## Philosophy: Configure, Don't Code

Every experiment parameter is controlled through a single YAML file. Researchers change settings by editing text, not by modifying Python code. This eliminates a major source of bugs and makes experiments fully reproducible.

## Configuration Sections

### 1. Experiment Metadata
```yaml
experiment:
  name: "pathology_v2"        # Used in output directory naming
  description: "DenseNet201 with augmentation"
  output_dir: "./experiments/"  # All outputs go here
```

### 2. Data Configuration
```yaml
data:
  train_data_path: "./data/train"
  test_data_path: "./data/test"
  val_data_path: null          # null → framework adapts (uses train metrics)
  num_classes: 2
  class_names: ["Normal", "Pathology"]
  image_size: 299
  format: "imagefolder"        # Auto-detected: "imagefolder" or "csv"
```

When `val_data_path` is null, the framework uses training metrics for model selection instead of validation metrics. When `train_data_path` is null, the framework enters **test-only mode**.

### 3. Preprocessing (11 Augmentation Toggles)
```yaml
preprocessing:
  normalize: true
  random_horizontal_flip: true
  random_rotation: 30
  color_jitter: false           # Toggle each independently
  # ... 8 more options
```

Each augmentation can be independently enabled/disabled. This gives full control without needing separate augmentation pipeline code.

### 4. Medical Imaging (DICOM/CLAHE)
```yaml
medical_imaging:
  enabled: true                 # Master toggle
  windowing:
    enabled: true
    window_center: -600
    window_width: 1500
  clahe:
    enabled: true
    clip_limit: 2.0
```

Disabled by default for standard image datasets. Enable only when working with DICOM files.

### 5. Model Selection (Weighted Multi-Metric)
```yaml
model_selection:
  strategy: "weighted"
  metrics:
    accuracy: 0.30
    f1_score: 0.30
    npv: 0.30
    auc_roc: 0.10
```

This replaced the naive "save model with highest accuracy" approach with a clinically-aware selection strategy.

## Config Parsing Architecture

Each nested YAML section maps to a `@dataclass` with a `from_dict()` class method:

```
config.yaml
├── experiment: → ExperimentConfig
├── data:       → DataConfig
├── model:      → ModelConfig
├── training:   → TrainingConfig
├── preprocessing: → PreprocessingConfig
├── medical_imaging:
│   ├── windowing: → WindowingConfig
│   └── clahe:     → CLAHEConfig
├── checkpoint:    → CheckpointConfig
├── model_selection: → ModelSelectionConfig
└── logging:       → LoggingConfig
```

All configs have sensible defaults, so a minimal YAML file needs only 3 fields: `train_data_path`, `test_data_path`, and `num_classes`.

## Total Configurable Options: 40+

| Category | # Options |
|----------|-----------|
| Data | 7 |
| Model | 5 |
| Training | 7 |
| Preprocessing / Augmentation | 11 |
| Medical Imaging | 5 |
| Checkpoint | 4 |
| Model Selection | 5 |
| Logging / Export | 7 |
| Reproducibility | 3 |
