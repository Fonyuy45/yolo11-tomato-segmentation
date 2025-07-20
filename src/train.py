#!/usr/bin/env python3
"""
YOLO11 Tomato Segmentation Training Script

This script trains a YOLO11 segmentation model on the Laboro Tomato dataset
for classifying tomato ripeness (green, half-ripened, fully-ripened).

Usage:
    python src/train.py --data data/data_config.yaml --model x --epochs 100
    python src/train.py --help

Author: Dieudonne Fonyuy Y.
Date: 2025
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch

def load_config(config_path):
    """Load training configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f" Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f" Error parsing YAML file: {e}")
        sys.exit(1)

def check_gpu():
    """Check GPU availability and print system info."""
    print(" System Information:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   GPU: Not available (using CPU)")
    print()

def train_model(config_path, model_size='x', epochs=100, batch_size=4, 
                imgsz=640, project='tomato_segmentation', name='training_run',
                resume=False, device=None):
    """
    Train YOLO11 segmentation model on tomato dataset.
    
    Args:
        config_path (str): Path to dataset configuration YAML
        model_size (str): Model size ('n', 's', 'm', 'l', 'x')
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        imgsz (int): Input image size
        project (str): Project name for saving results
        name (str): Experiment name
        resume (bool): Resume from last checkpoint
        device (str/int): Device to use for training
    
    Returns:
        YOLO: Trained model object
    """
    
    # Load and validate configuration
    print(" Loading configuration...")
    config = load_config(config_path)
    
    # Validate dataset paths
    dataset_path = Path(config['path'])
    if not dataset_path.exists():
        print(f" Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    print(f" Dataset found: {dataset_path}")
    print(f"   Classes: {config['names']}")
    print(f"   Number of classes: {config['nc']}")
    
    # Check GPU
    check_gpu()
    
    # Auto-detect device if not specified
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f" Loading YOLO11{model_size.upper()}-seg model...")
    model_name = f'yolo11{model_size}-seg.pt'
    model = YOLO(model_name)
    
    # Display model info
    print(f"   Model: {model_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Training configuration
    train_args = {
        'data': config_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': project,
        'name': name,
        'save': True,
        'plots': True,
        'val': True,
        'amp': True,  # Automatic Mixed Precision
        'cache': False,  # Don't cache images (save memory)
        'workers': min(8, os.cpu_count()),
        'patience': 100,  # Early stopping patience
        'save_period': 10,  # Save checkpoint every 10 epochs
    }
    
    # Add resume if specified
    if resume:
        train_args['resume'] = True
    
    print(f" Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {imgsz}")
    print(f"   Device: {device}")
    print(f"   Results will be saved to: {project}/{name}")
    print()
    
    try:
        # Train model
        results = model.train(**train_args)
        
        print(" Training completed successfully!")
        print(f"   Best model saved to: {project}/{name}/weights/best.pt")
        print(f"   Results saved to: {project}/{name}/")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("\n Final Metrics:")
            print(f"   Box mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
            print(f"   Box mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
            print(f"   Mask mAP@0.5: {metrics.get('metrics/mAP50(M)', 'N/A'):.3f}")
            print(f"   Mask mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(M)', 'N/A'):.3f}")
        
        return model
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n Training failed: {str(e)}")
        return None

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train YOLO11 segmentation model on tomato dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--data', type=str, default='data/data_config.yaml',
                       help='Path to dataset configuration YAML file')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='x', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (nano, small, medium, large, extra-large)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size (adjust based on GPU memory)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (pixels)')
    
    # Output arguments
    parser.add_argument('--project', type=str, default='tomato_segmentation',
                       help='Project name for saving results')
    parser.add_argument('--name', type=str, default='training_run',
                       help='Experiment name')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (0, 1, 2, ... or cpu)')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    # Convert device argument
    if args.device and args.device.isdigit():
        args.device = int(args.device)
    
    print(" YOLO11 Tomato Segmentation Training")
    print("=" * 50)
    
    # Train model
    model = train_model(
        config_path=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        resume=args.resume,
        device=args.device
    )
    
    if model is not None:
        print("\n Training script completed successfully!")
    else:
        print("\n Training script failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()