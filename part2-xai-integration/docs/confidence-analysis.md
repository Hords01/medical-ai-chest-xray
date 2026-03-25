# Confidence Analysis System

## Why Confidence Metrics Matter

A classification result without confidence is incomplete. "Pathology Present" could mean:
- The model is 99% sure → act on it
- The model is 52% sure → effectively a coin flip, get a second opinion

The confidence analysis system bridges this gap by providing quantitative and qualitative assessment of every prediction.

## Metrics Computed

### Confidence Score
The maximum softmax probability: `max(P(class_0), P(class_1))`

Ranges from 0.5 (maximum uncertainty in binary classification) to 1.0 (absolute certainty).

### Uncertainty
Simply `1 - confidence`. A direct measure of how unsure the model is.

### Shannon Entropy
`H = -Σ p_i × log(p_i)`

For binary classification:
- Minimum entropy = 0 (one class has probability 1.0)
- Maximum entropy = log(2) ≈ 0.693 (both classes equally likely)

Entropy captures a richer picture than raw confidence because it considers the full probability distribution, not just the maximum.

### Class Probabilities
The individual softmax outputs for each class, displayed as percentages. This lets physicians see exactly how the model weighted each option.

## Interpretation Categories

| Confidence | Category | Clinical Guidance |
|-----------|----------|-------------------|
| ≥ 95% | Çok Yüksek Güven | Model is very certain — result can be trusted with high confidence |
| ≥ 85% | Yüksek Güven | Model is quite certain — result is reliable |
| ≥ 70% | Orta Güven | Moderate certainty — consider in conjunction with clinical judgment |
| ≥ 55% | Düşük Güven | Model is uncertain — second opinion recommended |
| < 55% | Çok Düşük Güven | Model cannot decide — expert review required |

## Dual-Model Display

Both models' confidence metrics were displayed side-by-side in Tab 2 (İleri Analiz) of the Streamlit interface:

```
┌─────────────────────────┬─────────────────────────┐
│   Değerlendirme Modeli  │    Patoloji Modeli       │
├─────────────────────────┼─────────────────────────┤
│ Sınıf 0 Olasılığı: 8%  │ Sınıf 0 Olasılığı: 91% │
│ Sınıf 1 Olasılığı: 92% │ Sınıf 1 Olasılığı: 9%  │
│ Belirsizlik: 8%         │ Belirsizlik: 9%         │
│ Entropi: 0.3412         │ Entropi: 0.3687         │
│                         │                         │
│ Yüksek Güven:           │ Yüksek Güven:           │
│ Model oldukça emin      │ Model oldukça emin      │
└─────────────────────────┴─────────────────────────┘
```

This design allows physicians to independently assess trust in each classification axis — an image might be clearly suitable for evaluation (high evaluability confidence) but borderline for pathology (low pathology confidence).
