import os
import glob
import numpy as np

def yolo_bbox_to_polygon(class_id, x_center, y_center, width, height):
    """
    Convert YOLO format bounding box to polygon coordinates.
    
    Args:
        class_id: Class ID of the object
        x_center, y_center: Center coordinates of the bounding box (normalized)
        width, height: Width and height of the bounding box (normalized)
    
    Returns:
        List containing [class_id, x1, y1, x2, y2, x3, y3, x4, y4] where (xi, yi) are polygon vertices
    """
    # Calculate the corner points of the bounding box
    x1 = x_center - width/2
    y1 = y_center - height/2
    x2 = x_center + width/2
    y2 = y_center - height/2
    x3 = x_center + width/2
    y3 = y_center + height/2
    x4 = x_center - width/2
    y4 = y_center + height/2
    
    # Return polygon coordinates in clockwise order
    return [class_id, x1, y1, x2, y2, x3, y3, x4, y4]

def convert_yolo_file_to_polygons(input_file, output_file):
    """
    Convert a YOLO format annotation file to polygon format.
    
    Args:
        input_file: Path to input YOLO format file
        output_file: Path to output polygon format file
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    polygon_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:  # Standard YOLO format
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            polygon = yolo_bbox_to_polygon(class_id, x_center, y_center, width, height)
            polygon_line = ' '.join(map(str, polygon))
            polygon_lines.append(polygon_line)
    
    with open(output_file, 'w') as f:
        for line in polygon_lines:
            f.write(line + '\n')

def convert_dataset(input_dir, output_dir):
    """
    Convert all YOLO format files in a directory to polygon format.
    
    Args:
        input_dir: Directory containing YOLO format files
        output_dir: Directory to save polygon format files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for input_file in glob.glob(os.path.join(input_dir, '*.txt')):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        convert_yolo_file_to_polygons(input_file, output_file)
        print(f"Converted {filename}")

if __name__ == "__main__":
    # Example usage
    base_dir = "/Users/balaji.karrupuswamy/Downloads/Unet-CNN-main/datasets/labels"
    
    # Convert training data
    train_input_dir = os.path.join(base_dir, "train")
    train_output_dir = os.path.join(base_dir, "train")
    convert_dataset(train_input_dir, train_output_dir)
    
    # Convert validation data
    val_input_dir = os.path.join(base_dir, "val")
    val_output_dir = os.path.join(base_dir, "val")
    convert_dataset(val_input_dir, val_output_dir)
    
    print("Conversion complete!")