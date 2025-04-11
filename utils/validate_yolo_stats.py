import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import argparse
from collections import Counter
import pandas as pd
import seaborn as sns

def load_yaml(yaml_file):
    """Load YAML configuration file"""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def analyze_labels(label_dir, class_names, is_segment=False):
    """Analyze YOLO format labels and return statistics"""
    stats = {
        'class_distribution': Counter(),
        'sizes': [],
        'aspect_ratios': [],
        'positions': [],
        'classes': [],
        'points_per_polygon': [] if is_segment else None,
    }
    
    # Get all label files
    label_files = list(Path(label_dir).glob('*.txt'))
    print(f"Found {len(label_files)} label files in {label_dir}")
    
    if not label_files:
        print(f"No label files found in {label_dir}")
        return stats
    
    # Process each label file
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
            continue
        
        for line in lines:
            data = line.strip().split()
            if not data:
                continue
                
            try:
                class_id = int(data[0])
                stats['class_distribution'][class_id] += 1
                stats['classes'].append(class_id)
                
                if is_segment:
                    # Segmentation format: class_id x1 y1 x2 y2 ...
                    points = []
                    for i in range(1, len(data), 2):
                        if i + 1 < len(data):
                            x = float(data[i])
                            y = float(data[i + 1])
                            points.append((x, y))
                    
                    # Calculate bounding box from polygon
                    if points:
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        
                        width = max_x - min_x
                        height = max_y - min_y
                        center_x = (min_x + max_x) / 2
                        center_y = (min_y + max_y) / 2
                        
                        stats['sizes'].append((width, height))
                        stats['aspect_ratios'].append(width / height if height > 0 else 0)
                        stats['positions'].append((center_x, center_y))
                        stats['points_per_polygon'].append(len(points))
                else:
                    # Bounding box format: class_id center_x center_y width height
                    if len(data) == 5:
                        center_x, center_y = float(data[1]), float(data[2])
                        width, height = float(data[3]), float(data[4])
                        
                        stats['sizes'].append((width, height))
                        stats['aspect_ratios'].append(width / height if height > 0 else 0)
                        stats['positions'].append((center_x, center_y))
            except Exception as e:
                print(f"Error processing line '{line}' in {label_file}: {e}")
    
    return stats

def plot_statistics(stats, class_names, output_dir=None, is_segment=False):
    """Plot statistics from label analysis"""
    if not stats['classes']:
        print("No valid annotations found to plot statistics")
        return
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class Distribution
    plt.figure(figsize=(12, 6))
    class_counts = stats['class_distribution']
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    class_labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c in classes]
    
    plt.bar(class_labels, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    else:
        plt.show()
    
    # 2. Object Size Distribution
    plt.figure(figsize=(12, 6))
    sizes = np.array(stats['sizes'])
    
    if len(sizes) > 0:
        # Convert to area (width * height)
        areas = sizes[:, 0] * sizes[:, 1]
        
        plt.hist(areas, bins=50)
        plt.title('Object Size Distribution (Area)')
        plt.xlabel('Normalized Area')
        plt.ylabel('Count')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'size_distribution.png'))
        else:
            plt.show()
    
    # 3. Aspect Ratio Distribution
    plt.figure(figsize=(12, 6))
    aspect_ratios = np.array(stats['aspect_ratios'])
    
    if len(aspect_ratios) > 0:
        # Filter out extreme values for better visualization
        filtered_ratios = aspect_ratios[(aspect_ratios > 0.1) & (aspect_ratios < 10)]
        
        plt.hist(filtered_ratios, bins=50)
        plt.title('Aspect Ratio Distribution (width/height)')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Count')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'aspect_ratio_distribution.png'))
        else:
            plt.show()
    
    # 4. Spatial Distribution (Heatmap)
    plt.figure(figsize=(10, 10))
    positions = np.array(stats['positions'])
    
    if len(positions) > 0:
        # Create a 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1], 
            bins=20, range=[[0, 1], [0, 1]]
        )
        
        # Plot heatmap
        plt.imshow(heatmap.T, origin='lower', extent=[0, 1, 0, 1], 
                   aspect='equal', cmap='viridis')
        plt.colorbar(label='Count')
        plt.title('Spatial Distribution of Objects')
        plt.xlabel('X position (normalized)')
        plt.ylabel('Y position (normalized)')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'spatial_distribution.png'))
        else:
            plt.show()
    
    # 5. Points per Polygon (for segmentation only)
    if is_segment and stats['points_per_polygon']:
        plt.figure(figsize=(12, 6))
        plt.hist(stats['points_per_polygon'], bins=30)
        plt.title('Points per Polygon Distribution')
        plt.xlabel('Number of Points')
        plt.ylabel('Count')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'points_per_polygon.png'))
        else:
            plt.show()
    
    # 6. Class vs Size correlation
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame({
        'Class': [class_names[c] if c < len(class_names) else f"Class {c}" for c in stats['classes']],
        'Area': [w * h for w, h in stats['sizes']]
    })
    
    sns.boxplot(x='Class', y='Area', data=df)
    plt.title('Object Size by Class')
    plt.xlabel('Class')
    plt.ylabel('Area (normalized)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'size_by_class.png'))
    else:
        plt.show()

def validate_label_format(label_dir, is_segment=False):
    """Validate the format of YOLO labels"""
    label_files = list(Path(label_dir).glob('*.txt'))
    
    if not label_files:
        print(f"No label files found in {label_dir}")
        return False
    
    errors = []
    warnings = []
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                data = line.strip().split()
                if not data:
                    continue
                
                # Check class_id is an integer
                try:
                    class_id = int(data[0])
                    if class_id < 0:
                        errors.append(f"{label_file}, line {i+1}: Negative class ID {class_id}")
                except ValueError:
                    errors.append(f"{label_file}, line {i+1}: Class ID is not an integer")
                    continue
                
                # Check format based on type
                if is_segment:
                    # Segmentation format: class_id x1 y1 x2 y2 ...
                    if len(data) < 7 or len(data) % 2 != 1:  # At least 3 points (class_id + 3*2 coordinates)
                        errors.append(f"{label_file}, line {i+1}: Invalid segmentation format, need at least 3 points")
                        continue
                    
                    # Check all coordinates are valid floats between 0 and 1
                    for j in range(1, len(data)):
                        try:
                            coord = float(data[j])
                            if coord < 0 or coord > 1:
                                warnings.append(f"{label_file}, line {i+1}: Coordinate out of range [0,1]: {coord}")
                        except ValueError:
                            errors.append(f"{label_file}, line {i+1}: Invalid coordinate value: {data[j]}")
                else:
                    # Bounding box format: class_id center_x center_y width height
                    if len(data) != 5:
                        errors.append(f"{label_file}, line {i+1}: Invalid bounding box format, expected 5 values")
                        continue
                    
                    try:
                        center_x = float(data[1])
                        center_y = float(data[2])
                        width = float(data[3])
                        height = float(data[4])
                        
                        # Check coordinates are within range [0,1]
                        if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
                            warnings.append(f"{label_file}, line {i+1}: Center coordinates out of range [0,1]: ({center_x}, {center_y})")
                        
                        # Check width and height are positive and within range
                        if width <= 0 or height <= 0:
                            errors.append(f"{label_file}, line {i+1}: Width or height is not positive: {width}, {height}")
                        elif width > 1 or height > 1:
                            warnings.append(f"{label_file}, line {i+1}: Width or height > 1: {width}, {height}")
                    except ValueError:
                        errors.append(f"{label_file}, line {i+1}: Invalid numeric values for bounding box")
        
        except Exception as e:
            errors.append(f"Error processing {label_file}: {e}")
    
    # Report results
    if errors:
        print(f"Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    if warnings:
        print(f"Found {len(warnings)} warnings:")
        for warning in warnings[:10]:  # Show first 10 warnings
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    if not errors:
        print("No format errors found!")
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser(description='Validate and analyze YOLO format data')
    parser.add_argument('--yaml', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--type', type=str, default='detect', choices=['detect', 'segment'], 
                        help='Type of YOLO data (detect or segment)')
    parser.add_argument('--output', type=str, help='Output directory for statistics plots')
    args = parser.parse_args()
    
    # Load dataset configuration
    dataset_config = load_yaml(args.yaml)
    base_path = Path(os.path.dirname(args.yaml)) / dataset_config.get('path', '')
    class_names = dataset_config.get('names', [])
    
    # Get label directory
    val_dir = dataset_config.get('val', 'images/val')
    if 'images' in val_dir:
        val_label_dir = val_dir.replace('images', 'labels')
    else:
        val_label_dir = os.path.join('labels', val_dir)
    
    val_label_dir = os.path.join(base_path, val_label_dir)
    
    print(f"Validating YOLO {'segmentation' if args.type == 'segment' else 'detection'} labels in {val_label_dir}")
    
    # Validate label format
    is_valid = validate_label_format(val_label_dir, is_segment=(args.type == 'segment'))
    
    if is_valid or True:  # Continue with analysis even if there are format errors
        # Analyze labels
        stats = analyze_labels(val_label_dir, class_names, is_segment=(args.type == 'segment'))
        
        # Plot statistics
        plot_statistics(stats, class_names, args.output, is_segment=(args.type == 'segment'))

if __name__ == "__main__":
    main()