# Sample Images

This directory contains 8 representative images from the Laboro Tomato dataset.

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
