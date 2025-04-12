import torch
from ultralytics import YOLO
import os
from pathlib import Path

# The correct way to add safe globals for PyTorch 2.6+ compatibility
# We need to import the actual class, not just provide a string
from ultralytics.nn.tasks import SegmentationModel

torch.serialization.add_safe_globals(
    [(SegmentationModel.__module__, SegmentationModel.__name__)]
)

# Alternatively, you can use the register_safe_module approach
# import torch._weights_only_unpickler as wou
# wou.register_safe_module('ultralytics.nn.tasks')

model = YOLO("yolov8n-seg.pt")


# Verify dataset structure before training
# def verify_segmentation_dataset(base_path):
#     """Verify that the dataset follows the segmentation format requirements"""
#     base_path = Path(base_path)
#
#     # Check for required directories
#     for split in ['train', 'validate', 'test']:
#         images_dir = base_path / split / 'images'
#         labels_dir = base_path / split / 'labels'
#
#         if not images_dir.exists():
#             print(f"Warning: {images_dir} directory not found")
#
#         if not labels_dir.exists():
#             print(
#                 f"Warning: {labels_dir} directory not found. Segmentation requires mask labels.")
#             return False
#
#         # Check if labels contain polygon coordinates
#         if labels_dir.exists():
#             label_files = list(labels_dir.glob('*.txt'))
#             if label_files:
#                 with open(label_files[0], 'r') as f:
#                     line = f.readline().strip()
#                     values = line.split()
#                     if len(values) < 7:  # class_id + at least 3 points (6 coordinates)
#                         print(
#                             f"Warning: Label format may be incorrect for segmentation. Found {len(values)} values, expected at least 7.")
#                         print(
#                             "Segmentation requires polygon coordinates, not just bounding boxes.")
#                         return False
#
#     return True

script_dir = Path(__file__).parent.absolute()

dataset_path = os.path.join(script_dir, "datasets", "dataset.yaml")

# if verify_segmentation_dataset(os.path.join(script_dir,"datasets")):
#     print("Dataset structure looks valid for segmentation tasks.")
# else:
#     print("Dataset structure may not be valid for segmentation. Please check the format.")
#     print("See https://docs.ultralytics.com/datasets/segment/ for help.")

results = model.train(
    batch=8,
    device="cpu",
    data="C:\\Users\\Nakul Balaji\\OneDrive\\Documents\\GitHub\\DetechParticle\\datasets\\dataset.yaml",
    epochs=7,
    imgsz=120,
)



# results = model.predict('C:\\Users\\Nakul Balaji\\OneDrive\\Documents\\GitHub\\DetechParticle\\datasets\\images\\train\\train_0.jpg')
#
# import matplotlib as plt
# import cv2
#
# for result in results:
#     img_with_mask = result.plot()  # This returns an array (BGR by default)
#
#     # Convert BGR (OpenCV format) to RGB for matplotlib
#     img_rgb = cv2.cvtColor(img_with_mask, cv2.COLOR_BGR2RGB)
#
#     # Show image
#     plt.figure(figsize=(8, 8))
#     plt.imshow(img_rgb)
#     plt.title("YOLOv8 Segmentation Prediction")
#     plt.axis('off')
#     plt.show()