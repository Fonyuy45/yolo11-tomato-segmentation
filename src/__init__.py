"""
YOLO11 Tomato Segmentation Package

This package provides tools for training, inference, and evaluation of
YOLO11 models for tomato ripeness classification and segmentation.

Classes:
- Green tomatoes (unripe)
- Half-ripened tomatoes (partially ripe)  
- Fully-ripened tomatoes (ready for harvest)

Modules:
- train: Model training functionality
- predict: Inference and prediction
- evaluate: Model evaluation and metrics
- utils: Utility functions and helpers

Author: Dieudonne Fonyuy Y.
Date: 2025
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Dieudonne Fonyuy Y."
__email__ = "dieudonne.yufonyuy@gmail.com"

# Import main functions for easy access
try:
    from .train import train_model
    from .predict import predict_single_image, predict_batch
    from .evaluate import evaluate_model
except ImportError:
    # Handle case where dependencies are not installed
    pass

__all__ = [
    'train_model',
    'predict_single_image', 
    'predict_batch',
    'evaluate_model'
]