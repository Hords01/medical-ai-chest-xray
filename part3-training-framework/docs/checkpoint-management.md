# Checkpoint Management: 5 Strategies

## The Problem

Early versions of the training pipeline saved checkpoints aggressively: one for every "new best" and one for the latest epoch. A 100-epoch training run could produce 200+ checkpoint files, each hundreds of megabytes, consuming gigabytes of disk space.

For a lab running multiple experiments in parallel, this was unsustainable.

## The Solution: 5 Configurable Strategies

### Strategy 1: `none`
No checkpoints saved at all. Useful for quick experiments where you only care about final metrics, not the trained model itself.

**Files created:** 0

### Strategy 2: `best_only`
Saves only the single best model. When a new best is found, the previous best is deleted (if `cleanup_old_best: true`).

**Files created:** 1 (`best_model.pth`)

### Strategy 3: `best_and_latest` (Default)
Saves the best model AND the latest epoch. The "latest" file enables training resumption if interrupted, while "best" is the production model.

**Files created:** 2 (`best_model.pth`, `latest.pth`)

This is the default because it covers 90% of use cases: you get the best model for deployment and the ability to resume training — with minimal disk usage.

### Strategy 4: `every_n_epochs`
Saves a checkpoint every N epochs (configurable) plus the best model. Useful for analyzing how model performance evolves over time.

**Files created:** (epochs / N) + 1

### Strategy 5: `all`
Saves everything: best, latest, and every epoch. Full history for deep research analysis. Only recommended when disk space is not a concern.

**Files created:** epochs × 2 + 1

## Configuration

```yaml
checkpoint:
  strategy: "best_and_latest"
  save_only_simple_best: true      # Overwrite best_model.pth (no epoch suffix)
  save_only_simple_latest: true    # Overwrite latest.pth (no epoch suffix)
  cleanup_old_best: true           # Delete previous best when new best found
  # every_n: 10                    # Only for "every_n_epochs" strategy
```

## Disk Usage Comparison (100 Epochs, ~200MB per checkpoint)

| Strategy | Files | Disk Usage |
|----------|-------|------------|
| none | 0 | 0 |
| best_only | 1 | ~200MB |
| **best_and_latest** | **2** | **~400MB** |
| every_n (N=10) | 11 | ~2.2GB |
| all | 201 | ~40GB |

The default strategy reduced disk usage by **99%** compared to the original "save everything" approach.

## Checkpoint Contents

Each saved checkpoint includes:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,     # Model weights
    'optimizer_state_dict': OrderedDict,  # Optimizer state (for resumption)
    'score': float,                       # Composite selection score
    'metrics': dict,                      # All metrics at this epoch
}
```
