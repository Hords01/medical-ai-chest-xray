# Architecture Comparison: 26 CNN Models

## Architectures Tested

| Family | Models | Key Characteristic |
|--------|--------|--------------------|
| **AlexNet** | AlexNet | Pioneer CNN (2012), large FC layers, 57M params |
| **DenseNet** | 121, 161, 169, 201 | Dense connections — each layer receives input from all preceding layers |
| **EfficientNet** | EfficientNet-B0 | Compound scaling of depth/width/resolution |
| **GoogLeNet** | GoogLeNet | Inception modules with parallel convolution paths |
| **InceptionV3** | InceptionV3 | Factorized convolutions, auxiliary classifiers |
| **MobileNet** | V2, V3-large, V3-small | Depthwise separable convolutions for mobile deployment |
| **ResNet** | 18, 34, 50, 101, 152, 50-wide, 101-wide | Residual connections solving vanishing gradient |
| **ResNeXt** | 50-32×4d, 101-32×8d, 101-64×4d | Aggregated residual transformations (cardinality) |
| **VGG** | VGG16, VGG16-BN | Uniform 3×3 filters, simple and deep |
| **ShuffleNet** | V2-x0.5, x1.0, x1.5, x2.0 | Channel shuffle for efficient group convolutions |

## Why These Architectures?

The selection spans three generations of CNN design:

1. **Classical** (AlexNet, VGG): Establish baseline — raw depth and width
2. **Structural innovation** (ResNet, DenseNet, GoogLeNet): Skip connections, dense connections, inception modules
3. **Efficiency-focused** (MobileNet, ShuffleNet, EfficientNet): Same accuracy with 10-100× fewer parameters

This breadth allows us to answer: *For chest X-ray classification with limited data, does architectural sophistication translate to better results?*

## Key Findings

### 1. Parameter Count ≠ Performance

The correlation between parameter count and test accuracy is **weak**:

- **Best accuracy at 224×224:** ResNeXt101_64x4d (83.14%) — 81.4M params
- **Nearly identical:** MobileNetV3_small (82.49%) — 1.5M params
- **Worst performers include:** ResNet152 (73.02%) — 58.2M params, VGG16_bn (75.49%) — 134.3M params

With only 6,326 training images, large models overfit while compact models generalize.

### 2. Resolution Sensitivity

Switching from 224×224 to 299×299 changed rankings dramatically:

| Model | 224×224 | 299×299 | Change |
|-------|---------|---------|--------|
| GoogLeNet | 82.88% | **87.55%** | +4.67% |
| MobileNetV2 | 79.12% | 85.73% | +6.61% |
| ResNeXt101_64x4d | **83.14%** | 70.56% | -12.58% |
| ResNet152 | 73.02% | 69.78% | -3.24% |

**Interpretation:** Models with multi-scale feature extraction (Inception modules, depthwise separable convolutions) benefit from higher resolution. Very deep models with fixed receptive fields may struggle as input size increases without proportional depth.

### 3. Architecture Design > Depth

The DenseNet family illustrates this clearly:

| Model | Depth | Params | 224×224 Acc | NPV (Fine-tuned) |
|-------|-------|--------|-------------|-------------------|
| DenseNet121 | 121 layers | 6.96M | 80.93% | 83.02% |
| DenseNet161 | 161 layers | 26.5M | 75.36% | 51.04% |
| DenseNet169 | 169 layers | 12.5M | 74.97% | 48.98% |
| DenseNet201 | 201 layers | 18.1M | 77.95% | **90.48%** |

DenseNet121 (shallowest) consistently outperforms DenseNet161 and DenseNet169. DenseNet201 excels on fine-tuned NPV despite middling from-scratch accuracy. This suggests that **dense connectivity patterns** matter more than raw layer count.

### 4. The Mobile Architecture Surprise

MobileNetV3_small with just 1.52M parameters matched or exceeded models with 50-100× more parameters. This has practical implications: a clinical decision support tool doesn't need a data center — it could run on a hospital workstation or even a tablet.

## Methodology Notes

All comparisons used identical conditions:
- Same hyperparameters (Adam, lr=0.001, CrossEntropy, 10 epochs, batch=32)
- Same normalization (mean=0.5, std=0.5)
- Same data splits (train/val/test from Kaggle dataset)
- Same random seed for reproducibility

This **controlled experiment** design ensures performance differences reflect architectural properties, not tuning advantages.
