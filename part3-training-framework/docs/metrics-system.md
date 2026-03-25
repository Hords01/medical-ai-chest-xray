# Metrics System & NPV

## Why Standard ML Metrics Aren't Enough for Medical AI

Most ML frameworks report accuracy, precision, recall, and F1. For clinical applications, these are necessary but insufficient. The framework adds **Negative Predictive Value (NPV)** as a first-class metric because it answers the most critical clinical question: *"When the model says 'no pathology,' can we trust it?"*

## NPV (Negative Predictive Value)

```
NPV = TN / (TN + FN)
```

Where:
- **TN (True Negative):** Correctly identified as "no pathology"
- **FN (False Negative):** Incorrectly identified as "no pathology" (actually has pathology)

### Clinical Significance

| Scenario | Consequence |
|----------|------------|
| False Positive (FP) | Patient gets unnecessary follow-up — stressful but not dangerous |
| **False Negative (FN)** | **Sick patient is told they're healthy — diagnosis delayed, disease progresses** |

NPV directly measures protection against the most dangerous error type. A model with 95% NPV means that among patients classified as "no pathology," 95% truly have no pathology — only 5% are missed.

## Multi-Class NPV Calculation

For multi-class problems (e.g., 6 disease types), NPV is computed per-class and then averaged:

```python
for each class i:
    TN_i = total correct predictions NOT involving class i
    FN_i = samples of class i that were classified as something else
    NPV_i = TN_i / (TN_i + FN_i)

NPV_macro = mean(NPV_1, NPV_2, ..., NPV_K)
```

## Weighted Model Selection with NPV

The framework's model selection can weight NPV:

```yaml
model_selection:
  strategy: "weighted"
  metrics:
    accuracy: 0.30
    f1_score: 0.30
    npv: 0.30          # NPV gets significant weight
    auc_roc: 0.10
```

This means a model with slightly lower accuracy but higher NPV may be selected over a more "accurate" model — because in clinical practice, the safer model is better.

## AUC-ROC Support

The framework supports AUC-ROC for both binary and multi-class settings:

| Setting | Strategy | Averaging |
|---------|----------|-----------|
| Binary | Direct computation | N/A |
| Multi-class | One-vs-Rest (OvR) | Weighted |
| Multi-class | One-vs-One (OvO) | Macro |

## Export Formats

All metrics are automatically exported in three formats:

| Format | Best For |
|--------|----------|
| **JSON** | Programmatic access, dashboards, CI/CD pipelines |
| **CSV** | Spreadsheet analysis, epoch-by-epoch trends |
| **Excel** | Formatted tables for academic papers and presentations |

Additionally, confusion matrices are saved as PNG images for visual inspection.
