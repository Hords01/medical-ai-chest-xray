# Test-Only Mode

## Purpose

Sometimes you have a trained model and just want to evaluate it on new data — no retraining needed. Test-only mode enables this with a single config change.

## Activation

Set `train_data_path` to `null` while providing a `test_data_path`:

```yaml
data:
  train_data_path: null          # ← This triggers test-only mode
  test_data_path: "./data/test"
  num_classes: 2

test_only:
  checkpoint_path: "./models/best_model.pth"
  strict_loading: false
```

The framework auto-detects this configuration and skips the entire training loop.

## Flexible Checkpoint Loading

### strict=True (Default)
Model architecture must exactly match the saved weights. Every layer must be present with identical shapes.

### strict=False
Enables **partial loading**: loads all matching keys and reports discrepancies.

```
Loading checkpoint with strict=False
  Loaded: 347/350 keys matched
  Missing keys: ['classifier.weight', 'classifier.bias']
  Unexpected keys: ['fc.weight']
```

This handles common scenarios:
- Model architecture was updated after training (e.g., changed dropout rate)
- Classifier head was modified for a different number of classes
- Auxiliary classifiers were added/removed

## Validation Handling

In test-only mode, the framework suppresses the "no validation data" warning that would normally appear during training. The warning is irrelevant since no training occurs.

## Output

Test-only mode produces:
- All standard metrics (accuracy, precision, recall, F1, NPV, AUC-ROC)
- Confusion matrix
- Classification report
- Per-class breakdown (if enabled)

But does NOT produce:
- Epoch-by-epoch training curves
- TensorBoard logs
- Checkpoint files
- Training loss plots
