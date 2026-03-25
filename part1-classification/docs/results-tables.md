# Complete Experimental Results

## 1. Open-Source Dataset — 224×224 Resolution (From Scratch)

All models trained with identical hyperparameters: Adam optimizer, lr=0.001, CrossEntropyLoss, 10 epochs, batch_size=32. No pretrained weights.

| Model | Test Accuracy (%) | Parameters (M) | Precision | Recall | F1 Score |
|-------|-------------------|-----------------|-----------|--------|----------|
| ResNeXt101_64x4d | **83.14** | 81.4 | **0.8612** | **0.8314** | **0.8221** |
| GoogLeNet | 82.88 | 12.0 | 0.8486 | 0.8288 | 0.8145 |
| MobileNetV3_small | 82.49 | 1.52 | 0.8392 | 0.8249 | 0.8123 |
| MobileNetV3_large | 82.23 | 4.21 | 0.8302 | 0.8223 | 0.8158 |
| ResNet50 | 81.71 | 23.5 | 0.8589 | 0.8171 | 0.7936 |
| AlexNet | 81.58 | 57.0 | 0.8356 | 0.8158 | 0.8020 |
| DenseNet121 | 80.93 | 6.96 | 0.8502 | 0.8093 | 0.7870 |
| ResNeXt101_32x8d | 80.03 | 86.8 | 0.8515 | 0.8003 | 0.7755 |
| ShuffleNetV2_x2_0 | 80.03 | 5.35 | 0.8454 | 0.8003 | 0.7733 |
| EfficientNet | 79.51 | 4.01 | 0.8299 | 0.7951 | 0.7674 |
| VGG16 | 79.51 | 134.3 | 0.8363 | 0.7951 | 0.7757 |
| ResNet101 | 79.25 | 42.5 | 0.8389 | 0.7925 | 0.7611 |
| MobileNetV2 | 79.12 | 2.23 | 0.8364 | 0.7912 | 0.7683 |
| ShuffleNetV2_x0_5 | 78.21 | 0.35 | 0.8301 | 0.7821 | 0.7484 |
| DenseNet201 | 77.95 | 18.1 | 0.8376 | 0.7795 | 0.7403 |
| ResNeXt50_32x4d | 77.82 | 23.0 | 0.8341 | 0.7782 | 0.7422 |
| ResNet101_wide | 77.30 | 124.8 | 0.8186 | 0.7730 | 0.7402 |
| VGG16_bn | 75.49 | 134.3 | 0.8035 | 0.7549 | 0.7210 |
| DenseNet161 | 75.36 | 26.5 | 0.8120 | 0.7536 | 0.7150 |
| DenseNet169 | 74.97 | 12.5 | 0.8242 | 0.7497 | 0.7078 |
| ResNet34 | 74.71 | 21.3 | 0.8183 | 0.7471 | 0.6921 |
| ShuffleNetV2_x1_5 | 74.19 | 2.48 | 0.8089 | 0.7419 | 0.6868 |
| ShuffleNetV2_x1_0 | 73.93 | 1.26 | 0.8206 | 0.7393 | 0.6781 |
| ResNet152 | 73.02 | 58.2 | 0.8088 | 0.7302 | 0.6543 |
| ResNet50_wide | 70.95 | 66.8 | 0.8044 | 0.7095 | 0.6495 |
| ResNet18 | 68.48 | 11.2 | 0.8117 | 0.6848 | 0.6230 |

### Notable observations:
- **Parameter efficiency champion:** ShuffleNetV2_x0_5 with only 350K params achieved 78.21%
- **Diminishing returns:** ResNet152 (58.2M) scored lower than ResNet50 (23.5M) — classic overfitting on limited data
- **VGG16 vs VGG16_bn:** Batch normalization hurt performance here (75.49% vs 79.51%), possibly due to small batch size interactions

---

## 2. Open-Source Dataset — 299×299 Resolution (From Scratch)

Same conditions as above, with 299×299 input resolution. InceptionV3 added.

| Rank | Model | Test Accuracy (%) | Parameters (M) | Precision | Recall | F1 Score |
|------|-------|-------------------|-----------------|-----------|--------|----------|
| 1 | GoogLeNet | **87.55** | 11.98 | **0.8836** | **0.8755** | **0.8689** |
| 2 | DenseNet161 | 85.99 | 26.48 | 0.8717 | 0.8599 | 0.8532 |
| 3 | MobileNetV2 | 85.73 | 2.23 | 0.8767 | 0.8573 | 0.8476 |
| 4 | ShuffleNetV2_x1_0 | 84.05 | 1.26 | 0.8460 | 0.8405 | 0.8327 |
| 5 | ShuffleNetV2_x1_5 | 83.66 | 2.48 | 0.8523 | 0.8366 | 0.8261 |
| 6 | ResNet50 | 83.92 | 23.52 | 0.8585 | 0.8392 | 0.8289 |
| 7 | MobileNetV3_small | 83.66 | 1.52 | 0.8554 | 0.8366 | 0.8229 |
| 8 | InceptionV3 | 82.75 | 25.12 | 0.8628 | 0.8275 | 0.8084 |
| 9 | DenseNet169 | 82.88 | 12.49 | 0.8489 | 0.8288 | 0.8191 |
| 10 | VGG16_bn | 81.71 | 134.29 | 0.8373 | 0.8171 | 0.8049 |
| 11 | DenseNet201 | 81.71 | 18.10 | 0.8556 | 0.8171 | 0.8003 |
| 12 | VGG16 | 80.29 | 134.28 | 0.8461 | 0.8029 | 0.7806 |
| 13 | MobileNetV3_large | 80.16 | 4.21 | 0.8427 | 0.8016 | 0.7781 |
| 14 | ResNet101 | 78.34 | 42.51 | 0.8172 | 0.7834 | 0.7610 |
| 15 | ResNet50_wide | 78.21 | 66.84 | 0.8163 | 0.7821 | 0.7569 |
| 16 | ResNet101_wide | 76.91 | 124.85 | 0.8214 | 0.7691 | 0.7397 |
| 17 | EfficientNet | 76.52 | 4.01 | 0.8107 | 0.7652 | 0.7259 |
| 18 | DenseNet121 | 75.75 | 6.96 | 0.8262 | 0.7575 | 0.7107 |
| 19 | ResNeXt50_32x4d | 75.36 | 22.99 | 0.8242 | 0.7536 | 0.7090 |
| 20 | AlexNet | 74.58 | 57.02 | 0.8146 | 0.7458 | 0.7185 |
| 21 | ResNet18 | 73.28 | 11.18 | 0.8180 | 0.7328 | 0.6655 |
| 22 | ShuffleNetV2_x0_5 | 73.15 | 0.35 | 0.8021 | 0.7315 | 0.6698 |
| 23 | ResNeXt101_32x8d | 71.98 | 86.75 | 0.8068 | 0.7198 | 0.6592 |
| 24 | ShuffleNetV2_x2_0 | 71.34 | 5.35 | 0.8033 | 0.7134 | 0.6371 |
| 25 | ResNet34 | 71.85 | 21.29 | 0.8026 | 0.7185 | 0.6508 |
| 26 | ResNeXt101_64x4d | 70.56 | 81.41 | 0.7944 | 0.7056 | 0.6420 |
| 27 | ResNet152 | 69.78 | 58.15 | 0.7812 | 0.6978 | 0.6122 |

### Resolution impact highlights:
- **GoogLeNet:** 82.88% → 87.55% (+4.67%) — biggest winner
- **ResNeXt101_64x4d:** 83.14% → 70.56% (-12.58%) — biggest loser, likely overfitting to larger input
- **MobileNetV2:** 79.12% → 85.73% (+6.61%) — substantial gain

---

## 3. Ethics-Approved Dataset — Pathology Model (Without Fine-Tuning)

ImageNet1K_V1 pretrained, all layers frozen, only classifier retrained. 5 epochs, lr=0.001.

| Rank | Model | Test Accuracy (%) | NPV (%) | Precision (%) | Recall (%) | F1 Score |
|------|-------|-------------------|---------|---------------|------------|----------|
| 1 | DenseNet121 | 68.97 | **83.02** | 73.18 | 68.97 | 0.6708 |
| 2 | DenseNet201 | 71.92 | 74.12 | 72.16 | 71.92 | 0.7174 |
| 3 | ResNet50 | 74.38 | 72.12 | 74.52 | 74.38 | 0.7439 |
| 4 | EfficientNet | 73.40 | 68.33 | 74.74 | 73.40 | 0.7318 |
| 5 | InceptionV3 | 58.13 | 53.59 | 75.24 | 58.13 | 0.5067 |
| 6 | DenseNet161 | 53.69 | 51.04 | 76.36 | 53.69 | 0.4244 |
| 7 | ResNet101 | 49.75 | 49.00 | 75.38 | 49.75 | 0.3463 |
| 8 | DenseNet169 | 49.75 | 48.98 | 60.59 | 49.75 | 0.3615 |

---

## 4. Ethics-Approved Dataset — Evaluability Model (Without Fine-Tuning)

| Rank | Model | Test Accuracy (%) | NPV (%) | Precision (%) | Recall (%) | F1 Score |
|------|-------|-------------------|---------|---------------|------------|----------|
| 1 | ResNet101 | 66.50 | **77.69** | 68.41 | 66.50 | 0.6715 |
| 2 | DenseNet121 | 37.44 | 75.00 | 61.34 | 37.44 | 0.2738 |
| 3 | EfficientNet | 67.49 | 74.82 | 67.03 | 67.49 | 0.6723 |
| 4 | DenseNet201 | 66.50 | 70.81 | 63.84 | 66.50 | 0.6401 |
| 5 | ResNet50 | 69.46 | 70.17 | 67.98 | 69.46 | 0.6388 |
| 6 | DenseNet169 | 62.56 | 70.07 | 60.95 | 62.56 | 0.6155 |
| 7 | DenseNet161 | 65.02 | 66.67 | 56.52 | 65.02 | 0.5546 |
| 8 | InceptionV3 | 48.28 | 64.42 | 53.33 | 48.28 | 0.4972 |

---

## 5. Ethics-Approved Dataset — Pathology Model (With Fine-Tuning)

7 epochs, lr=0.0001, with data augmentation (CenterCrop, RandomRotation, RandomResizedCrop, RandomHorizontalFlip, Padding).

| Rank | Model | Test Accuracy (%) | NPV (%) | Precision (%) | Recall (%) | F1 Score |
|------|-------|-------------------|---------|---------------|------------|----------|
| 1 | DenseNet201 | 76.85 | **90.48** | 80.25 | 76.85 | 0.7598 |
| 2 | EfficientNet | 78.82 | 75.23 | 79.24 | 78.82 | 0.7880 |
| 3 | ResNet101 | **79.31** | 75.00 | **79.97** | **79.31** | **0.7926** |
| 4 | DenseNet121 | 75.37 | 72.22 | 75.70 | 75.37 | 0.7535 |
| 5 | ResNet50 | 76.35 | 71.55 | 77.35 | 76.35 | 0.7624 |
| 6 | InceptionV3 | 75.37 | 69.67 | 77.06 | 75.37 | 0.7512 |
| 7 | DenseNet169 | 71.92 | 67.52 | 72.89 | 71.92 | 0.7177 |
| 8 | DenseNet161 | 70.94 | 65.60 | 72.78 | 70.94 | 0.7055 |

---

## 6. Ensemble Voting Results (Final System)

Models: DenseNet201 + EfficientNet + ResNet101 (fine-tuned on pathology task)

| Method | Accuracy (%) | NPV (%) | Precision (%) | Recall (%) | F1 Score |
|--------|-------------|---------|---------------|------------|----------|
| Majority Voting | 84.03 | 82.28 | 86.15 | 80.00 | 0.8296 |
| **Soft Voting** | **86.11** | **87.50** | 84.72 | **87.14** | **0.8592** |
| **Weighted Soft Voting** | **86.11** | **87.50** | 84.72 | **87.14** | **0.8592** |
| Thresholded Soft Voting | 81.25 | 75.82 | **90.57** | 68.57 | 0.7805 |

> Note: Thresholded voting achieves highest precision (90.57%) at the cost of sensitivity — useful when false alarms are expensive but not ideal for screening.
