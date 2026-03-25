# Ensemble Voting: Combining Three Models

## Why Ensemble?

Each model has different strengths:

| Model | Strength | Weakness |
|-------|----------|----------|
| **DenseNet201** | Highest NPV (90.48%) — excellent at identifying "no pathology" | Lower overall accuracy (76.85%) |
| **EfficientNet** | Balanced performance across all metrics | No standout metric |
| **ResNet101** | Highest accuracy (79.31%) and F1 (0.7926) | Lower NPV (75.00%) |

Ensemble methods exploit these complementary strengths: where DenseNet201 catches true negatives, ResNet101 catches true positives.

## Four Strategies Compared

### 1. Majority Voting
Each model casts a hard vote (argmax). Class with ≥2 votes wins.

**Pros:** Simple, interpretable, no hyperparameters
**Cons:** Loses probability information — a 51% prediction counts the same as a 99% prediction

**Result:** 84.03% accuracy, 82.28% NPV

### 2. Soft Voting ⭐
Average the softmax probabilities across all models. Highest average wins.

**Pros:** Uses full probability distribution, handles uncertain predictions gracefully
**Cons:** Assumes models are equally reliable

**Result:** 86.11% accuracy, 87.50% NPV, 0.8592 F1

### 3. Weighted Soft Voting ⭐
Same as soft voting, but weighted by individual model performance.

Weights (normalized from test accuracy):
- DenseNet201: 0.3268
- EfficientNet: 0.3352
- ResNet101: 0.3380

**Result:** 86.11% accuracy, 87.50% NPV, 0.8592 F1

Identical to soft voting because the models had similar enough performance that weighting made negligible difference.

### 4. Thresholded Soft Voting
Only accept predictions where the averaged confidence exceeds a threshold.

**Result:** 81.25% accuracy, 75.82% NPV, but 90.57% precision

This strategy is the most conservative. It catches fewer cases overall but is very rarely wrong when it does predict pathology. Useful in scenarios where false alarms are costly — but not ideal for screening where missing a sick patient is worse.

## Why Soft Voting Won

The key insight is that probability averaging **smooths out individual model errors**. When DenseNet201 says "85% no pathology" but ResNet101 says "60% pathology," the averaged probability reflects this genuine uncertainty rather than forcing a binary choice.

For clinical screening, this means the system is both more accurate overall AND more reliable at identifying truly healthy patients (high NPV).

## Improvement Over Individual Models

| Metric | Best Individual | Soft Voting Ensemble | Improvement |
|--------|----------------|---------------------|-------------|
| Accuracy | 79.31% (ResNet101) | 86.11% | **+6.80%** |
| NPV | 90.48% (DenseNet201) | 87.50% | -2.98%* |
| F1 Score | 0.7926 (ResNet101) | 0.8592 | **+0.0666** |
| Sensitivity | 79.31% (ResNet101) | 87.14% | **+7.83%** |

*NPV slightly decreased from DenseNet201's individual 90.48%, but the ensemble's NPV is significantly higher than the other two models' individual NPV (~75%). The ensemble trades a small NPV decrease from the best model for massive gains in all other metrics.
