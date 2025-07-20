# YOLO11 Tomato Segmentation Dataset Configuration
# Laboro Tomato Dataset - Roboflow Universe

# Dataset paths (relative to this file)
path: ../datasets/Laboro_Tomato
train: train/images
val: valid/images
test: test/images  # optional

# Number of classes
nc: 3

# Class names (in order of class indices)
names:
  0: fully_ripened
  1: green  
  2: half_ripened

# Class descriptions
class_descriptions:
  fully_ripened: "Fully ripe tomatoes ready for harvest (red color)"
  green: "Unripe green tomatoes (green color)"
  half_ripened: "Partially ripe tomatoes (transitioning color)"

# Dataset information
dataset_info:
  name: "Laboro Tomato Dataset"
  version: "v2 (Second_2024-02-04 3-20pm)"
  source: "Roboflow Universe"
  url: "https://universe.roboflow.com/jalals-lab/laboro-tomato-kwpth/dataset/2"
  generated: "February 4, 2024"
  license: "CC BY 4.0"
  total_images: 2174
  train_images: 2001
  val_images: 86
  test_images: 87
  total_instances: 898
  annotation_format: "YOLO segmentation masks"

# Roboflow preprocessing applied
preprocessing:
  auto_orient: true
  resize: "Stretch to 640x640"
  class_modifications: "6 remapped, 0 dropped"
  filter_null: true

# Roboflow augmentations applied
roboflow_augmentations:
  outputs_per_example: 3
  flip: ["horizontal", "vertical"]
  rotation: "-15° to +15°"
  shear: "±10° horizontal, ±10° vertical"
  blur: "Up to 1.5px"
  noise: "Up to 1.56% of pixels"

# Training configuration
train_config:
  model_size: "x"  # n, s, m, l, x
  epochs: 100
  batch_size: 4
  image_size: 640
  device: 0  # GPU device
  workers: 8
  
# Augmentation settings
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0
  copy_paste: 0.0

# Class distribution (instances per class)
class_distribution:
  fully_ripened: 163
  green: 568  
  half_ripened: 167

# Performance metrics (from best model)
performance:
  box_map50: 0.901
  box_map50_95: 0.805
  mask_map50: 0.898
  mask_map50_95: 0.771
  inference_speed_ms: 28
