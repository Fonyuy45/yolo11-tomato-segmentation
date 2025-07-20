#!/usr/bin/env python3
"""
Utility functions for YOLO11 Tomato Segmentation

This module provides helper functions for data processing, visualization,
and model utilities.

Author: Dieudonne Fonyuy Y.
Date: 2025
"""

import os
import yaml
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import torch
from ultralytics import YOLO

# Class names and colors for visualization
CLASS_NAMES = {
    0: 'fully_ripened',
    1: 'green', 
    2: 'half_ripened'
}

CLASS_COLORS = {
    'fully_ripened': (0, 0, 255),    # Red
    'green': (0, 255, 0),            # Green
    'half_ripened': (0, 165, 255)   # Orange
}

def load_yaml_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to YAML file
        
    Returns:
        Dict: Loaded configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

def save_json(data: Dict, filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data (Dict): Data to save
        filepath (str): Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def check_gpu_memory() -> Dict[str, float]:
    """
    Check GPU memory usage.
    
    Returns:
        Dict: GPU memory information
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated_memory = torch.cuda.memory_allocated(0) / 1e9
    cached_memory = torch.cuda.memory_reserved(0) / 1e9
    free_memory = total_memory - allocated_memory
    
    return {
        "available": True,
        "total_gb": total_memory,
        "allocated_gb": allocated_memory,
        "cached_gb": cached_memory,
        "free_gb": free_memory,
        "utilization_percent": (allocated_memory / total_memory) * 100
    }

def visualize_predictions(image_path: str, results, save_path: Optional[str] = None,
                         show_conf: bool = True, show_labels: bool = True) -> np.ndarray:
    """
    Visualize YOLO predictions on an image.
    
    Args:
        image_path (str): Path to input image
        results: YOLO results object
        save_path (str, optional): Path to save visualization
        show_conf (bool): Whether to show confidence scores
        show_labels (bool): Whether to show class labels
        
    Returns:
        np.ndarray: Visualization image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                class_name = CLASS_NAMES.get(int(cls), f"Class_{int(cls)}")
                color = CLASS_COLORS.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                label_parts = []
                if show_labels:
                    label_parts.append(class_name)
                if show_conf:
                    label_parts.append(f"{conf:.2f}")
                
                if label_parts:
                    label = " ".join(label_parts)
                    
                    # Get text size for background
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        image_rgb, 
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        color, -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        image_rgb, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )
        
        # Draw segmentation masks if available
        if hasattr(r, 'masks') and r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            
            for mask, cls in zip(masks, classes):
                class_name = CLASS_NAMES.get(int(cls), f"Class_{int(cls)}")
                color = CLASS_COLORS.get(class_name, (255, 255, 255))
                
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
                mask_colored = np.zeros_like(image_rgb)
                mask_colored[mask_resized > 0.5] = color
                
                # Blend mask with image
                image_rgb = cv2.addWeighted(image_rgb, 0.8, mask_colored, 0.3, 0)
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image_bgr)
    
    return image_rgb

def calculate_class_distribution(results_list: List) -> Dict[str, int]:
    """
    Calculate class distribution from prediction results.
    
    Args:
        results_list (List): List of YOLO results
        
    Returns:
        Dict: Class distribution counts
    """
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    
    for results in results_list:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                classes = r.boxes.cls.cpu().numpy()
                for cls in classes:
                    class_name = CLASS_NAMES.get(int(cls), f"Class_{int(cls)}")
                    if class_name in class_counts:
                        class_counts[class_name] += 1
    
    return class_counts

def plot_class_distribution(class_counts: Dict[str, int], save_path: Optional[str] = None) -> None:
    """
    Plot class distribution as a bar chart.
    
    Args:
        class_counts (Dict): Class counts
        save_path (str, optional): Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [CLASS_COLORS.get(cls, (0.5, 0.5, 0.5)) for cls in classes]
    # Convert BGR to RGB and normalize for matplotlib
    colors = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
    
    bars = ax.bar(classes, counts, color=colors)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{count}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Tomato Class Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Ripeness Class', fontsize=12)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def validate_model_compatibility(model_path: str, expected_classes: int = 3) -> bool:
    """
    Validate that a model is compatible with the expected task.
    
    Args:
        model_path (str): Path to model file
        expected_classes (int): Expected number of classes
        
    Returns:
        bool: True if compatible
    """
    try:
        model = YOLO(model_path)
        
        # Check if it's a segmentation model
        if not hasattr(model.model, 'model') or not hasattr(model.model.model[-1], 'seg'):
            print("  Warning: Model may not be a segmentation model")
            return False
        
        # Check number of classes
        if hasattr(model.model, 'yaml'):
            model_classes = model.model.yaml.get('nc', 0)
            if model_classes != expected_classes:
                print(f"  Warning: Model has {model_classes} classes, expected {expected_classes}")
                return False
        
        return True
        
    except Exception as e:
        print(f" Error validating model: {e}")
        return False

def create_inference_summary(results_list: List, processing_times: List[float]) -> Dict:
    """
    Create a summary of inference results.
    
    Args:
        results_list (List): List of YOLO results
        processing_times (List): Processing times for each image
        
    Returns:
        Dict: Summary statistics
    """
    total_detections = 0
    total_images = len(results_list)
    class_counts = calculate_class_distribution(results_list)
    confidence_scores = []
    
    for results in results_list:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                total_detections += len(r.boxes)
                confidence_scores.extend(r.boxes.conf.cpu().numpy())
    
    summary = {
        'total_images': total_images,
        'total_detections': total_detections,
        'average_detections_per_image': total_detections / total_images if total_images > 0 else 0,
        'class_distribution': class_counts,
        'confidence_stats': {
            'mean': float(np.mean(confidence_scores)) if confidence_scores else 0.0,
            'std': float(np.std(confidence_scores)) if confidence_scores else 0.0,
            'min': float(np.min(confidence_scores)) if confidence_scores else 0.0,
            'max': float(np.max(confidence_scores)) if confidence_scores else 0.0,
        },
        'timing_stats': {
            'mean_ms': float(np.mean(processing_times)) if processing_times else 0.0,
            'std_ms': float(np.std(processing_times)) if processing_times else 0.0,
            'min_ms': float(np.min(processing_times)) if processing_times else 0.0,
            'max_ms': float(np.max(processing_times)) if processing_times else 0.0,
            'fps': 1000 / np.mean(processing_times) if processing_times and np.mean(processing_times) > 0 else 0.0
        }
    }
    
    return summary