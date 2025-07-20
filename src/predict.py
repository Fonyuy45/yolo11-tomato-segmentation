#!/usr/bin/env python3
"""
YOLO11 Tomato Segmentation Inference Script

This script runs inference on images/videos using a trained YOLO11 model
to classify tomato ripeness and generate segmentation masks.

Usage:
    python src/predict.py --model models/best.pt --source data/sample_images/
    python src/predict.py --model models/best.pt --source image.jpg --save-txt
    python src/predict.py --help

Author: Dieudonne Fonyuy Y.
Date: 2025
"""

import argparse
import os
import sys
from pathlib import Path
import time
from ultralytics import YOLO
import cv2
import numpy as np

def load_model(model_path):
    """Load YOLO11 model from checkpoint."""
    try:
        model = YOLO(model_path)
        print(f" Model loaded: {model_path}")
        
        # Print model info
        if hasattr(model.model, 'yaml'):
            print(f"   Classes: {model.model.names}")
            print(f"   Model type: Segmentation")
        
        return model
    except Exception as e:
        print(f" Failed to load model: {e}")
        sys.exit(1)

def predict_single_image(model, image_path, conf_threshold=0.25, save_dir=None):
    """
    Run inference on a single image.
    
    Args:
        model: YOLO model object
        image_path: Path to input image
        conf_threshold: Confidence threshold for predictions
        save_dir: Directory to save results
    
    Returns:
        Results object with predictions
    """
    start_time = time.time()
    
    # Run inference
    results = model(image_path, conf=conf_threshold)
    
    inference_time = time.time() - start_time
    
    # Print results
    print(f"\n Image: {Path(image_path).name}")
    print(f"   Inference time: {inference_time*1000:.1f}ms")
    
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            print(f"   Detections: {len(r.boxes)}")
            
            # Print class counts
            if hasattr(r.boxes, 'cls'):
                classes = r.boxes.cls.cpu().numpy()
                class_names = [model.names[int(c)] for c in classes]
                from collections import Counter
                class_counts = Counter(class_names)
                for class_name, count in class_counts.items():
                    print(f"     {class_name}: {count}")
        else:
            print("   No detections")
    
    # Save results if save_dir specified
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, r in enumerate(results):
            output_path = save_path / f"{Path(image_path).stem}_result.jpg"
            r.save(filename=str(output_path))
            print(f"   Saved: {output_path}")
    
    return results

def predict_batch(model, source_dir, conf_threshold=0.25, save_dir=None, 
                 save_txt=False, save_conf=False):
    """
    Run batch inference on multiple images.
    
    Args:
        model: YOLO model object
        source_dir: Directory containing images
        conf_threshold: Confidence threshold
        save_dir: Directory to save results
        save_txt: Save results as text files
        save_conf: Save confidence scores
    
    Returns:
        List of results objects
    """
    source_path = Path(source_dir)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = [f for f in source_path.rglob('*') 
                      if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f" No image files found in {source_dir}")
        return []
    
    print(f" Found {len(image_files)} images")
    
    all_results = []
    total_time = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        start_time = time.time()
        
        # Run inference
        results = model(
            str(image_path), 
            conf=conf_threshold,
            save=bool(save_dir),
            save_txt=save_txt,
            save_conf=save_conf,
            project=save_dir,
            name='predictions',
            exist_ok=True
        )
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Print quick stats
        for r in results:
            detections = len(r.boxes) if r.boxes is not None else 0
            print(f"   Detections: {detections}, Time: {inference_time*1000:.1f}ms")
        
        all_results.extend(results)
    
    # Print summary
    avg_time = total_time / len(image_files)
    print(f"\n Batch Summary:")
    print(f"   Images processed: {len(image_files)}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time per image: {avg_time*1000:.1f}ms")
    print(f"   FPS: {1/avg_time:.1f}")
    
    return all_results

def analyze_results(results, model):
    """Analyze and summarize prediction results."""
    total_detections = 0
    class_counts = {}
    confidence_scores = []
    
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            total_detections += len(r.boxes)
            
            # Count classes
            if hasattr(r.boxes, 'cls'):
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                
                for cls, conf in zip(classes, confs):
                    class_name = model.names[int(cls)]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    confidence_scores.append(conf)
    
    print(f"\n Overall Analysis:")
    print(f"   Total detections: {total_detections}")
    
    if class_counts:
        print(f"   Class distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / total_detections) * 100
            print(f"     {class_name}: {count} ({percentage:.1f}%)")
    
    if confidence_scores:
        avg_conf = np.mean(confidence_scores)
        print(f"   Average confidence: {avg_conf:.3f}")
        print(f"   Confidence range: {min(confidence_scores):.3f} - {max(confidence_scores):.3f}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run YOLO11 tomato segmentation inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image file or directory')
    
    # Inference arguments
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for predictions')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='results/predictions',
                       help='Directory to save prediction results')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save results as text files')
    parser.add_argument('--save-conf', action='store_true',
                       help='Save confidence scores in text files')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save prediction images')
    
    # Display arguments
    parser.add_argument('--show', action='store_true',
                       help='Display results (for single images)')
    parser.add_argument('--analyze', action='store_true',
                       help='Print detailed analysis of results')
    
    args = parser.parse_args()
    
    print("🍅 YOLO11 Tomato Segmentation Inference")
    print("=" * 50)
    
    # Load model
    model = load_model(args.model)
    
    # Check source
    source_path = Path(args.source)
    if not source_path.exists():
        print(f" Source not found: {args.source}")
        sys.exit(1)
    
    # Determine save directory
    save_dir = None if args.no_save else args.save_dir
    
    # Run inference
    if source_path.is_file():
        # Single image
        results = predict_single_image(
            model, args.source, args.conf, save_dir
        )
        
        if args.show and results:
            results[0].show()
    else:
        # Batch processing
        results = predict_batch(
            model, args.source, args.conf, save_dir, 
            args.save_txt, args.save_conf
        )
    
    # Analyze results
    if args.analyze and results:
        analyze_results(results, model)
    
    print("\n Inference completed!")

if __name__ == "__main__":
    main()