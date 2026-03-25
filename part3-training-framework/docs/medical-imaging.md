# Medical Imaging Preprocessing

## Why Standard Image Preprocessing Isn't Enough

Medical images — especially DICOM format from CT/MR scanners — are fundamentally different from natural photographs:

1. **Hounsfield Units (HU):** DICOM pixel values range from -1000 to +3000, representing tissue density. Standard images use 0-255.
2. **Tissue-specific visualization:** Different anatomical structures require different value ranges to be visible.
3. **Low contrast:** X-ray images often have poor contrast, especially in portable/bedside acquisitions.

## DICOM Windowing

### The Problem

A raw DICOM image displayed linearly maps 4000+ HU values to 256 gray levels. This means each gray level represents ~16 HU — far too coarse to distinguish between tissues that differ by only a few HU.

### The Solution: Windowing

Windowing selects a narrow range of HU values and maps only that range to 0-255:

```
window_min = window_center - window_width / 2
window_max = window_center + window_width / 2

pixel_value = (HU_value - window_min) / (window_max - window_min) × 255
clip to [0, 255]
```

### Common Presets

| Tissue Type | Center | Width | Visible Structures |
|-------------|--------|-------|-------------------|
| **Lung** | -600 | 1500 | Airways, parenchyma, nodules |
| **Soft Tissue** | 40 | 400 | Organs, muscles, vessels |
| **Bone** | 400 | 1800 | Skeletal structures, calcifications |
| **Brain** | 40 | 80 | Gray/white matter differentiation |
| **Liver** | 60 | 150 | Hepatic lesions, vessels |

For chest X-ray pathology detection, the **lung window** (center=-600, width=1500) was the primary choice, as it maximizes visibility of pulmonary structures.

## CLAHE (Contrast Limited Adaptive Histogram Equalization)

### The Problem with Global Histogram Equalization

Standard histogram equalization adjusts the entire image uniformly. In medical images, this can:
- Over-brighten already well-exposed regions
- Amplify noise in dark regions
- Create unrealistic contrast jumps at tissue boundaries

### How CLAHE Works

1. Divide the image into small tiles (e.g., 8×8 grid)
2. Compute histogram equalization locally within each tile
3. Apply a **clip limit** to prevent over-amplification of noise
4. Bilinear interpolation between tiles eliminates boundary artifacts

```yaml
clahe:
  enabled: true
  clip_limit: 2.0           # Higher = more contrast, but more noise
  tile_grid_size: [8, 8]    # Smaller tiles = more local adaptation
```

### Clinical Impact

CLAHE significantly improves visibility of:
- Subtle ground-glass opacities (COVID-19, interstitial lung disease)
- Small nodules against rib structures
- Pleural effusions in low-contrast portable X-rays

## Configuration

Both preprocessing steps are fully optional and toggled via YAML:

```yaml
medical_imaging:
  enabled: false            # Master switch — false for standard JPEG/PNG datasets
  windowing:
    enabled: false
    window_center: -600
    window_width: 1500
  clahe:
    enabled: false
    clip_limit: 2.0
    tile_grid_size: [8, 8]
```

This means the framework works equally well for non-medical image datasets — just leave `medical_imaging.enabled: false`.
