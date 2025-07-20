#!/usr/bin/env python3
"""
Download sample images from the Laboro Tomato dataset for demonstration.
This script creates a sample_images directory with representative images.
"""

import os
import shutil
import random
from pathlib import Path

def create_sample_data():
    """Create sample images directory with representative examples."""
    
    # Paths
    dataset_path = Path("../datasets/Laboro_Tomato")
    sample_dir = Path("sample_images")
    
    # Create sample directory
    sample_dir.mkdir(exist_ok=True)
    
    # Check if dataset exists
    if not dataset_path.exists():
        print("Dataset not found. Please ensure Laboro_Tomato dataset is downloaded.")
        print(f"Expected path: {dataset_path.absolute()}")
        return False
    
    # Get train images
    train_images = list((dataset_path / "train" / "images").glob("*.jpg"))
    
    if not train_images:
        print("No training images found in dataset.")
        return False
    
    # Select random sample images (5-10 images)
    num_samples = min(8, len(train_images))
    sample_images = random.sample(train_images, num_samples)
    
    print(f"📸 Copying {num_samples} sample images...")
    
    # Copy sample images
    for i, img_path in enumerate(sample_images):
        dest_path = sample_dir / f"tomato_sample_{i+1}.jpg"
        shutil.copy2(img_path, dest_path)
        print(f"Copied: {img_path.name} -> {dest_path.name}")
    
    # Create a README for samples
    readme_content = f"""# Sample Images

This directory contains {num_samples} representative images from the Laboro Tomato dataset.

## Usage

These images can be used for:
- Quick testing of the trained model
- Demonstration purposes
- README documentation

## Run Inference

```bash
# Single image prediction
python src/predict.py --model models/best.pt --source data/sample_images/tomato_sample_1.jpg

# Batch prediction on all samples
python src/predict.py --model models/best.pt --source data/sample_images/
```

## Original Dataset

Full dataset available at: https://universe.roboflow.com/jalals-lab/laboro-tomato-kwpth/dataset/2
"""
    
    with open(sample_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"Sample data created successfully in {sample_dir}/")
    print(f"Created README.md with usage instructions")
    
    return True

if __name__ == "__main__":
    print("🍅 Creating sample data for YOLO11 Tomato Segmentation...")
    success = create_sample_data()
    
    if success:
        print("\n Sample data setup complete!")
        print("You can now use these images for quick testing and demonstrations.")
    else:
        print("\n Failed to create sample data.")
        print("Please ensure the Laboro_Tomato dataset is properly downloaded.")
