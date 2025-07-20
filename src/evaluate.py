#!/usr/bin/env python3
"""
YOLO11 Tomato Segmentation Model Evaluation Script

This script evaluates a trained YOLO11 model on the validation dataset
and generates comprehensive performance metrics and visualizations.

Usage:
    python src/evaluate.py --model models/best.pt --data data/data_config.yaml
    python src/evaluate.py --model models/best.pt --data data/data_config.yaml --save-plots
    python src/evaluate.py --help

Author: Dieudonne Fonyuy Y.
Date: 2025
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import json
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_config(config_path):
    """Load dataset configuration from YAML file."""
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

def evaluate_model(model_path, data_config, save_dir=None, imgsz=640, 
                  conf_threshold=0.001, iou_threshold=0.6, save_plots=False):
    """
    Evaluate YOLO11 model on validation dataset.
    
    Args:
        model_path (str): Path to trained model
        data_config (str): Path to dataset configuration
        save_dir (str): Directory to save evaluation results
        imgsz (int): Input image size
        conf_threshold (float): Confidence threshold for evaluation
        iou_threshold (float): IoU threshold for NMS
        save_plots (bool): Whether to save additional plots
    
    Returns:
        dict: Evaluation results
    """
    
    # Load model
    print(f" Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        print(f" Model loaded successfully")
    except Exception as e:
        print(f" Failed to load model: {e}")
        sys.exit(1)
    
    # Load dataset config
    config = load_config(data_config)
    print(f" Dataset: {config.get('dataset_info', {}).get('name', 'Unknown')}")
    print(f"   Classes: {list(config['names'].values())}")
    
    # Run validation
    print(f" Running validation...")
    print(f"   Image size: {imgsz}")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   IoU threshold: {iou_threshold}")
    
    try:
        # Validate model
        results = model.val(
            data=data_config,
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            save_json=True,
            save_hybrid=False,
            plots=save_plots,
            project=save_dir,
            name='evaluation',
            exist_ok=True
        )
        
        print(" Validation completed successfully!")
        
    except Exception as e:
        print(f" Validation failed: {e}")
        sys.exit(1)
    
    return results, config

def extract_metrics(results):
    """Extract key metrics from validation results."""
    metrics = {}
    
    # Box detection metrics
    if hasattr(results, 'box'):
        box_metrics = results.box
        metrics['box'] = {
            'map50': float(box_metrics.map50) if hasattr(box_metrics, 'map50') else 0.0,
            'map50_95': float(box_metrics.map) if hasattr(box_metrics, 'map') else 0.0,
            'precision': float(box_metrics.mp) if hasattr(box_metrics, 'mp') else 0.0,
            'recall': float(box_metrics.mr) if hasattr(box_metrics, 'mr') else 0.0,
        }
    
    # Segmentation metrics
    if hasattr(results, 'seg'):
        seg_metrics = results.seg
        metrics['segmentation'] = {
            'map50': float(seg_metrics.map50) if hasattr(seg_metrics, 'map50') else 0.0,
            'map50_95': float(seg_metrics.map) if hasattr(seg_metrics, 'map') else 0.0,
            'precision': float(seg_metrics.mp) if hasattr(seg_metrics, 'mp') else 0.0,
            'recall': float(seg_metrics.mr) if hasattr(seg_metrics, 'mr') else 0.0,
        }
    
    # Per-class metrics
    if hasattr(results, 'box') and hasattr(results.box, 'ap_class_index'):
        class_indices = results.box.ap_class_index
        class_ap50 = results.box.ap50
        
        metrics['per_class'] = {}
        for i, class_idx in enumerate(class_indices):
            metrics['per_class'][int(class_idx)] = {
                'box_ap50': float(class_ap50[i]) if i < len(class_ap50) else 0.0
            }
    
    # Speed metrics
    if hasattr(results, 'speed'):
        speed = results.speed
        metrics['speed'] = {
            'preprocess_ms': float(speed.get('preprocess', 0)),
            'inference_ms': float(speed.get('inference', 0)),
            'postprocess_ms': float(speed.get('postprocess', 0)),
        }
        metrics['speed']['total_ms'] = sum(metrics['speed'].values())
    
    return metrics

def generate_report(metrics, config, model_path, save_dir):
    """Generate a comprehensive evaluation report."""
    
    # Create report content
    report = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'dataset': config.get('dataset_info', {}).get('name', 'Unknown'),
            'dataset_version': config.get('dataset_info', {}).get('version', 'Unknown'),
            'num_classes': config['nc'],
            'class_names': config['names']
        },
        'metrics': metrics
    }
    
    if save_dir:
        # Save JSON report
        save_path = Path(save_dir) / 'evaluation'
        save_path.mkdir(parents=True, exist_ok=True)
        
        json_path = save_path / 'evaluation_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f" Report saved: {json_path}")
        
        # Save human-readable report
        txt_path = save_path / 'evaluation_summary.txt'
        with open(txt_path, 'w') as f:
            f.write("YOLO11 Tomato Segmentation - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Model: {model_path}\n")
            f.write(f"Dataset: {config.get('dataset_info', {}).get('name', 'Unknown')}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Box detection metrics
            if 'box' in metrics:
                f.write("Box Detection Metrics:\n")
                f.write(f"  mAP@0.5: {metrics['box']['map50']:.3f}\n")
                f.write(f"  mAP@0.5:0.95: {metrics['box']['map50_95']:.3f}\n")
                f.write(f"  Precision: {metrics['box']['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['box']['recall']:.3f}\n\n")
            
            # Segmentation metrics
            if 'segmentation' in metrics:
                f.write("Segmentation Metrics:\n")
                f.write(f"  mAP@0.5: {metrics['segmentation']['map50']:.3f}\n")
                f.write(f"  mAP@0.5:0.95: {metrics['segmentation']['map50_95']:.3f}\n")
                f.write(f"  Precision: {metrics['segmentation']['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['segmentation']['recall']:.3f}\n\n")
            
            # Per-class metrics
            if 'per_class' in metrics:
                f.write("Per-Class Performance:\n")
                class_names = config['names']
                for class_idx, class_metrics in metrics['per_class'].items():
                    class_name = class_names.get(class_idx, f"Class_{class_idx}")
                    f.write(f"  {class_name}: {class_metrics['box_ap50']:.3f}\n")
                f.write("\n")
            
            # Speed metrics
            if 'speed' in metrics:
                f.write("Speed Performance:\n")
                f.write(f"  Preprocessing: {metrics['speed']['preprocess_ms']:.1f}ms\n")
                f.write(f"  Inference: {metrics['speed']['inference_ms']:.1f}ms\n")
                f.write(f"  Postprocessing: {metrics['speed']['postprocess_ms']:.1f}ms\n")
                f.write(f"  Total: {metrics['speed']['total_ms']:.1f}ms\n")
                f.write(f"  FPS: {1000/metrics['speed']['total_ms']:.1f}\n")
        
        print(f" Summary saved: {txt_path}")
    
    return report

def print_results(metrics, config):
    """Print evaluation results to console."""
    
    print("\n EVALUATION RESULTS")
    print("=" * 50)
    
    # Box detection results
    if 'box' in metrics:
        print("\n Object Detection Performance:")
        print(f"   mAP@0.5: {metrics['box']['map50']:.3f}")
        print(f"   mAP@0.5:0.95: {metrics['box']['map50_95']:.3f}")
        print(f"   Precision: {metrics['box']['precision']:.3f}")
        print(f"   Recall: {metrics['box']['recall']:.3f}")
    
    # Segmentation results
    if 'segmentation' in metrics:
        print("\n Segmentation Performance:")
        print(f"   mAP@0.5: {metrics['segmentation']['map50']:.3f}")
        print(f"   mAP@0.5:0.95: {metrics['segmentation']['map50_95']:.3f}")
        print(f"   Precision: {metrics['segmentation']['precision']:.3f}")
        print(f"   Recall: {metrics['segmentation']['recall']:.3f}")
    
    # Per-class results
    if 'per_class' in metrics:
        print("\n Per-Class Performance:")
        class_names = config['names']
        for class_idx, class_metrics in metrics['per_class'].items():
            class_name = class_names.get(class_idx, f"Class_{class_idx}")
            print(f"   {class_name}: {class_metrics['box_ap50']:.3f}")
    
    # Speed results
    if 'speed' in metrics:
        print("\n⚡ Speed Performance:")
        print(f"   Preprocessing: {metrics['speed']['preprocess_ms']:.1f}ms")
        print(f"   Inference: {metrics['speed']['inference_ms']:.1f}ms") 
        print(f"   Postprocessing: {metrics['speed']['postprocess_ms']:.1f}ms")
        print(f"   Total: {metrics['speed']['total_ms']:.1f}ms")
        print(f"   FPS: {1000/metrics['speed']['total_ms']:.1f}")

def create_visualization_plots(metrics, config, save_dir):
    """Create additional visualization plots."""
    if not save_dir:
        return
    
    save_path = Path(save_dir) / 'evaluation'
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Per-class performance plot
    if 'per_class' in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        class_names = []
        ap_scores = []
        
        for class_idx, class_metrics in metrics['per_class'].items():
            class_name = config['names'].get(class_idx, f"Class_{class_idx}")
            class_names.append(class_name)
            ap_scores.append(class_metrics['box_ap50'])
        
        bars = ax.bar(class_names, ap_scores)
        ax.set_title('Per-Class Average Precision (AP@0.5)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Precision', fontsize=12)
        ax.set_xlabel('Tomato Ripeness Class', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, ap_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Color bars based on performance
        for bar, score in zip(bars, ap_scores):
            if score >= 0.9:
                bar.set_color('green')
            elif score >= 0.8:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Per-class plot saved: {save_path}/per_class_performance.png")
    
    # Performance comparison plot
    if 'box' in metrics and 'segmentation' in metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box vs Segmentation mAP comparison
        metrics_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
        box_values = [
            metrics['box']['map50'], 
            metrics['box']['map50_95'],
            metrics['box']['precision'],
            metrics['box']['recall']
        ]
        seg_values = [
            metrics['segmentation']['map50'],
            metrics['segmentation']['map50_95'], 
            metrics['segmentation']['precision'],
            metrics['segmentation']['recall']
        ]
        
        x = range(len(metrics_names))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], box_values, width, label='Box Detection', alpha=0.8)
        ax1.bar([i + width/2 for i in x], seg_values, width, label='Segmentation', alpha=0.8)
        
        ax1.set_title('Detection vs Segmentation Performance', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_xlabel('Metric', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names, rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Speed breakdown pie chart
        if 'speed' in metrics:
            speed_data = metrics['speed']
            labels = ['Preprocessing', 'Inference', 'Postprocessing']
            sizes = [
                speed_data['preprocess_ms'],
                speed_data['inference_ms'], 
                speed_data['postprocess_ms']
            ]
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Inference Time Breakdown', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Comparison plot saved: {save_path}/performance_comparison.png")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO11 tomato segmentation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset configuration YAML file')
    
    # Evaluation arguments
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='Confidence threshold for evaluation')
    parser.add_argument('--iou', type=float, default=0.6,
                       help='IoU threshold for NMS')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save additional visualization plots')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip saving detailed report files')
    
    args = parser.parse_args()
    
    print("🍅 YOLO11 Tomato Segmentation Evaluation")
    print("=" * 50)
    
    # Check inputs
    if not Path(args.model).exists():
        print(f" Model file not found: {args.model}")
        sys.exit(1)
    
    if not Path(args.data).exists():
        print(f" Data config file not found: {args.data}")
        sys.exit(1)
    
    # Run evaluation
    results, config = evaluate_model(
        args.model, args.data, args.save_dir, args.imgsz,
        args.conf, args.iou, args.save_plots
    )
    
    # Extract metrics
    metrics = extract_metrics(results)
    
    # Print results
    print_results(metrics, config)
    
    # Generate report
    if not args.no_report:
        report = generate_report(metrics, config, args.model, args.save_dir)
    
    # Create additional plots
    if args.save_plots:
        create_visualization_plots(metrics, config, args.save_dir)
    
    print("\n Evaluation completed successfully!")

if __name__ == "__main__":
    main()