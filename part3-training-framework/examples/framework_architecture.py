"""
ML Training Framework — Class Architecture
=============================================
Illustrative overview of the framework's modular class design.

The framework follows a separation-of-concerns pattern:
each class handles one responsibility, configured via YAML.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import yaml


# ============================================================
# 1. CONFIGURATION CLASSES
# ============================================================

class CheckpointStrategy(Enum):
    NONE = "none"
    BEST_ONLY = "best_only"
    BEST_AND_LATEST = "best_and_latest"
    EVERY_N_EPOCHS = "every_n_epochs"
    ALL = "all"


@dataclass
class WindowingConfig:
    """DICOM Hounsfield Unit windowing configuration."""
    enabled: bool = False
    window_center: float = -600.0    # Default: lung window
    window_width: float = 1500.0

    @classmethod
    def from_dict(cls, d: dict) -> 'WindowingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CLAHEConfig:
    """Contrast Limited Adaptive Histogram Equalization."""
    enabled: bool = False
    clip_limit: float = 2.0
    tile_grid_size: List[int] = field(default_factory=lambda: [8, 8])

    @classmethod
    def from_dict(cls, d: dict) -> 'CLAHEConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MedicalImagingConfig:
    """Container for all medical imaging preprocessing options."""
    enabled: bool = False
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    clahe: CLAHEConfig = field(default_factory=CLAHEConfig)

    @classmethod
    def from_dict(cls, d: dict) -> 'MedicalImagingConfig':
        return cls(
            enabled=d.get('enabled', False),
            windowing=WindowingConfig.from_dict(d.get('windowing', {})),
            clahe=CLAHEConfig.from_dict(d.get('clahe', {})),
        )


@dataclass
class RandomAffineConfig:
    """Random affine transformation settings."""
    enabled: bool = False
    degrees: float = 10.0
    translate: Optional[List[float]] = None
    scale: Optional[List[float]] = None
    shear: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'RandomAffineConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentConfig:
    """
    Master configuration — single source of truth for the entire pipeline.

    Loaded from YAML, validated, and passed to all components.
    40+ configurable options, all with sensible defaults.
    """
    # Experiment
    name: str = "experiment"
    description: str = ""
    output_dir: str = "./experiments/"

    # Data
    train_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    num_classes: int = 2
    class_names: Optional[List[str]] = None
    image_size: int = 224
    data_format: str = "imagefolder"

    # Model
    architecture: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = True
    dropout_rate: float = 0.0

    # Training
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    criterion: str = "cross_entropy"

    # Checkpoint
    checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.BEST_AND_LATEST
    cleanup_old_best: bool = True

    # Medical Imaging
    medical_imaging: MedicalImagingConfig = field(default_factory=MedicalImagingConfig)

    # Logging
    tensorboard: bool = True
    confusion_matrix: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    auto_readme: bool = True

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        config = cls()

        # Flatten nested YAML into config attributes
        if 'experiment' in raw:
            config.name = raw['experiment'].get('name', config.name)
            config.description = raw['experiment'].get('description', config.description)
            config.output_dir = raw['experiment'].get('output_dir', config.output_dir)

        if 'data' in raw:
            d = raw['data']
            config.train_data_path = d.get('train_data_path')
            config.test_data_path = d.get('test_data_path')
            config.val_data_path = d.get('val_data_path')
            config.num_classes = d.get('num_classes', config.num_classes)
            config.class_names = d.get('class_names')
            config.image_size = d.get('image_size', config.image_size)
            config.data_format = d.get('format', config.data_format)

        if 'medical_imaging' in raw:
            config.medical_imaging = MedicalImagingConfig.from_dict(raw['medical_imaging'])

        # ... (other sections parsed similarly)

        return config

    @property
    def is_test_only(self) -> bool:
        """Auto-detect test-only mode when no training data is provided."""
        return self.train_data_path is None and self.test_data_path is not None


# ============================================================
# 2. CHECKPOINT MANAGER
# ============================================================

class CheckpointManager:
    """
    Manages model checkpoint saving with 5 configurable strategies.

    Problem solved: Without management, 100 epochs could create
    200+ checkpoint files consuming gigabytes. Default strategy
    keeps only 2 files: best_model.pth and latest.pth.
    """

    def __init__(self, config: ExperimentConfig):
        self.strategy = config.checkpoint_strategy
        self.output_dir = config.output_dir
        self.cleanup = config.cleanup_old_best
        self.best_score = -float('inf')
        self.best_path = None

    def should_save(self, epoch: int, score: float) -> Dict[str, bool]:
        """
        Determine what to save based on strategy.

        Returns dict with keys: 'save_best', 'save_latest', 'save_epoch'
        """
        is_best = score > self.best_score

        if self.strategy == CheckpointStrategy.NONE:
            return {'save_best': False, 'save_latest': False, 'save_epoch': False}

        elif self.strategy == CheckpointStrategy.BEST_ONLY:
            return {'save_best': is_best, 'save_latest': False, 'save_epoch': False}

        elif self.strategy == CheckpointStrategy.BEST_AND_LATEST:
            return {'save_best': is_best, 'save_latest': True, 'save_epoch': False}

        elif self.strategy == CheckpointStrategy.EVERY_N_EPOCHS:
            # Save every N epochs (configured separately)
            return {'save_best': is_best, 'save_latest': False, 'save_epoch': True}

        elif self.strategy == CheckpointStrategy.ALL:
            return {'save_best': is_best, 'save_latest': True, 'save_epoch': True}

        return {'save_best': False, 'save_latest': False, 'save_epoch': False}

    def save(self, model, optimizer, epoch, score, metrics):
        """Save checkpoint(s) according to strategy."""
        actions = self.should_save(epoch, score)

        if actions['save_best']:
            if self.cleanup and self.best_path:
                # Remove previous best to save disk space
                import os
                if os.path.exists(self.best_path):
                    os.remove(self.best_path)

            self.best_score = score
            self.best_path = f"{self.output_dir}/best_model.pth"
            self._save_checkpoint(model, optimizer, epoch, score, metrics, self.best_path)

        if actions['save_latest']:
            path = f"{self.output_dir}/latest.pth"
            self._save_checkpoint(model, optimizer, epoch, score, metrics, path)

        if actions['save_epoch']:
            path = f"{self.output_dir}/epoch_{epoch:04d}.pth"
            self._save_checkpoint(model, optimizer, epoch, score, metrics, path)

    def _save_checkpoint(self, model, optimizer, epoch, score, metrics, path):
        import torch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'metrics': metrics,
        }, path)


# ============================================================
# 3. MODEL SELECTION
# ============================================================

class ModelSelector:
    """
    Smart model selection using weighted multi-metric scoring.

    Instead of just picking the highest accuracy, this combines
    multiple metrics with configurable weights. Critical for
    medical applications where NPV matters more than accuracy.
    """

    def __init__(self, metric_weights: Dict[str, float]):
        self.weights = metric_weights
        self.best_score = -float('inf')
        self.best_epoch = -1

    def compute_score(self, metrics: Dict[str, float]) -> float:
        """
        Weighted combination of metrics.

        Example with default weights:
        score = 0.40 * accuracy + 0.30 * f1 + 0.20 * npv + 0.10 * auc_roc
        """
        score = 0.0
        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                score += weight * metrics[metric_name]
        return score

    def is_best(self, metrics: Dict[str, float], epoch: int) -> bool:
        score = self.compute_score(metrics)
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            return True
        return False


# ============================================================
# 4. METRICS CALCULATOR
# ============================================================

class MetricsCalculator:
    """
    Comprehensive metrics including NPV, AUC-ROC, and per-class breakdown.

    NPV (Negative Predictive Value) was added specifically for this project
    because standard ML frameworks don't include it, yet it's the most
    critical metric for clinical screening applications.
    """

    @staticmethod
    def compute_all(y_true, y_pred, y_probs=None, num_classes=2) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, roc_auc_score
        )

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # NPV calculation
        cm = confusion_matrix(y_true, y_pred)
        if num_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        else:
            # Multi-class NPV: per-class then average
            npvs = []
            for i in range(num_classes):
                # For class i: TN = sum of all entries not in row i or col i
                tn_i = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
                fn_i = cm[i, :].sum() - cm[i, i]
                npvs.append(tn_i / (tn_i + fn_i) if (tn_i + fn_i) > 0 else 0.0)
            metrics['npv'] = float(np.mean(npvs))

        # AUC-ROC (if probabilities available)
        if y_probs is not None:
            try:
                if num_classes == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    metrics['auc_roc'] = roc_auc_score(
                        y_true, y_probs,
                        multi_class='ovr', average='weighted'
                    )
            except ValueError:
                metrics['auc_roc'] = 0.0

        return metrics


# ============================================================
# 5. USAGE PATTERN
# ============================================================

def main():
    """
    Complete training flow with the framework.

    In practice: python train.py --config my_config.yaml
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    # Auto-detect mode
    if config.is_test_only:
        print("Test-only mode: evaluating existing model")
        # Load checkpoint and run evaluation only
        return

    # Initialize components
    checkpoint_mgr = CheckpointManager(config)
    model_selector = ModelSelector(metric_weights={
        'accuracy': 0.30, 'f1_score': 0.30, 'npv': 0.30, 'auc_roc': 0.10
    })
    metrics_calc = MetricsCalculator()

    # Training loop (simplified)
    # for epoch in range(config.epochs):
    #     train_metrics = train_one_epoch(...)
    #     val_metrics = evaluate(...)
    #
    #     # Smart model selection
    #     selection_metrics = val_metrics if config.val_data_path else train_metrics
    #     is_best = model_selector.is_best(selection_metrics, epoch)
    #
    #     # Checkpoint management
    #     score = model_selector.compute_score(selection_metrics)
    #     checkpoint_mgr.save(model, optimizer, epoch, score, selection_metrics)

    print(f"Training complete. Best model at epoch {model_selector.best_epoch}")


if __name__ == "__main__":
    main()
