# Dataset Documentation

## Overview

The YOLO11 Tomato Segmentation project uses the **Laboro Tomato Dataset** from Roboflow Universe, specifically designed for instance segmentation of tomatoes at different ripeness stages.

## Dataset Information

- **Name**: Laboro Tomato Dataset
- **Version**: v2 (Second_2024-02-04 3-20pm)
- **Source**: [Roboflow Universe](https://universe.roboflow.com/jalals-lab/laboro-tomato-kwpth/dataset/2)
- **Generated**: February 4, 2024
- **License**: CC BY 4.0
- **Format**: YOLO segmentation masks
- **Total Images**: 2,174
- **Annotation Quality**: High-quality polygonal segmentation masks

## Dataset Split

| Split | Images | Percentage |
|-------|---------|------------|
| **Training** | 2,001 | 92% |
| **Validation** | 86 | 4% |
| **Test** | 87 | 4% |
| **Total** | 2,174 | 100% |

## Classes and Distribution

The dataset contains **3 classes** representing different tomato ripeness stages:

### Class Definitions

| Class ID | Class Name | Description | Color Code | Instances |
|----------|------------|-------------|------------|-----------|
| 0 | `fully_ripened` | Ready-to-harvest red tomatoes | 🔴 Red | 163 |
| 1 | `green` | Unripe green tomatoes | 🟢 Green | 568 |
| 2 | `half_ripened` | Partially ripe transition tomatoes | 🟡 Orange | 167 |

### Distribution Analysis

- **Total Instances**: 898 tomato instances
- **Most Common**: Green tomatoes (63.2%)
- **Least Common**: Fully ripened tomatoes (18.2%)
- **Average Instances per Image**: ~4.3 tomatoes

## Data Quality

### Annotation Quality
- ✅ **High precision polygonal masks** for instance segmentation
- ✅ **Consistent labeling** across ripeness stages
- ✅ **Diverse lighting conditions** and angles
- ✅ **Multiple tomatoes per image** (realistic scenarios)

### Image Characteristics
- **Resolution**: 640x640 (stretched from variable resolutions)
- **Format**: JPEG images
- **Preprocessing**: Auto-orient applied, stretched to 640x640
- **Class Remapping**: 6 classes remapped, 0 dropped
- **Filter**: All images contain annotations (null filtered)
- **Environments**: Greenhouse and outdoor settings
- **Lighting**: Natural and artificial lighting conditions
- **Backgrounds**: Typical agricultural environments with foliage

## Dataset Preparation

### Roboflow Preprocessing Applied

The dataset has been preprocessed with the following transformations:
- **Auto-Orient**: Applied to correct image orientation
- **Resize**: Stretch to 640x640 pixels
- **Modify Classes**: 6 classes remapped, 0 classes dropped
- **Filter Null**: Only images with annotations included

### Augmentations Applied

During dataset creation, the following augmentations were used:
- **Outputs per training example**: 3 (3x data augmentation)
- **Flip**: Horizontal and Vertical flipping
- **Rotation**: Between -15° and +15°
- **Shear**: ±10° Horizontal, ±10° Vertical
- **Blur**: Up to 1.5px blur effect
- **Noise**: Up to 1.56% of pixels with noise

## Dataset Preparation

### Download Instructions

```bash
# Using Roboflow (recommended)
pip install roboflow
python scripts/download_dataset.py

# Manual download from Roboflow Universe
# Visit: https://universe.roboflow.com/jalals-lab/laboro-tomato-kwpth/dataset/2
```

### Directory Structure

```
datasets/Laboro_Tomato/
├── train/
│   ├── images/          # Training images (2,001 files)
│   └── labels/          # Training segmentation masks (2,001 files)
├── valid/
│   ├── images/          # Validation images (86 files)
│   └── labels/          # Validation segmentation masks (86 files)
├── test/
│   ├── images/          # Test images (87 files)
│   └── labels/          # Test segmentation masks (87 files)
└── data.yaml           # Dataset configuration
```

### Label Format

The dataset uses YOLO segmentation format:
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

Where:
- `class_id`: 0 (fully_ripened), 1 (green), or 2 (half_ripened)
- `x1, y1, ..., xn, yn`: Normalized polygon coordinates (0-1)

## Usage in Training

### Configuration

The dataset is configured in `data/data_config.yaml`:

```yaml
path: ../datasets/Laboro_Tomato
train: train/images
val: valid/images
nc: 3
names:
  0: fully_ripened
  1: green
  2: half_ripened
```

### Training Command

```bash
python src/train.py --data data/data_config.yaml --model x --epochs 100 --batch 4
```

## Performance Benchmarks

### Achieved Results on This Dataset

| Metric | Box Detection | Instance Segmentation |
|--------|---------------|----------------------|
| **mAP@0.5** | 90.1% | 89.8% |
| **mAP@0.5:0.95** | 80.5% | 77.1% |

### Per-Class Performance

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|---------|---------|
| **Fully Ripened** | 91.4% | 91.4% | 91.3% |
| **Green** | 86.8% | 91.8% | 91.8% |
| **Half Ripened** | 85.0% | 87.1% | 87.1% |

## Data Augmentation

The following augmentations were applied during training:

- **Geometric**: Horizontal flip (50%), scaling, translation
- **Color**: HSV augmentation (hue ±1.5%, saturation ±70%, value ±40%)
- **Mosaic**: 4-image mosaic augmentation (100% probability)
- **Mixup**: Disabled for this dataset
- **Copy-Paste**: Disabled for this dataset

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{laboro_tomato_2024,
  title={Laboro Tomato Dataset},
  author={Jalal's Lab},
  year={2024},
  publisher={Roboflow Universe},
  url={https://universe.roboflow.com/jalals-lab/laboro-tomato-kwpth/dataset/2},
  license={CC BY 4.0}
}
```

## Related Datasets

For additional tomato detection datasets, consider:
- **Tomato Detection Dataset** (Roboflow)
- **Plant Disease Recognition Dataset** 
- **Agricultural Object Detection Datasets**

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure the dataset is downloaded to `datasets/Laboro_Tomato/`
2. **Permission errors**: Check file permissions after download
3. **Corrupted files**: Re-download the dataset if validation fails

### Verification

```bash
# Verify dataset structure
python data/download_sample_data.py

# Check dataset statistics
python -c "
import yaml
with open('data/data_config.yaml') as f:
    config = yaml.safe_load(f)
print(f'Classes: {config[\"names\"]}')
print(f'Number of classes: {config[\"nc\"]}')
"
```
