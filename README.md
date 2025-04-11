# YOLO Object Detection with Ultralytics

This repository contains scripts for training and using YOLO models for bounding box detection using the Ultralytics library.

## Dataset Structure

The dataset is organized in the YOLO format:
```
datasets/
├── dataset.yaml    # Dataset configuration
├── train/
│   ├── images/     # Training images
│   └── labels/     # Training annotations
├── validate/
│   ├── images/     # Validation images
│   └── labels/     # Validation annotations
└── test/
    ├── images/     # Test images
    └── labels/     # Test annotations
```

## Training a YOLO Model

To train a YOLO model, use the `train_yolo.py` script:

```bash
python train_yolo.py --epochs 100 --batch 16 --img-size 640 --model yolov8n.pt --data datasets/dataset.yaml
```

### Training Arguments

- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--img-size`: Input image size (default: 640)
- `--device`: Device to use (empty for auto-selection)
- `--model`: Pretrained model to start from (default: yolov8n.pt)
- `--data`: Path to dataset configuration file (default: datasets/dataset.yaml)
- `--project`: Project name for saving results (default: runs/detect)
- `--name`: Experiment name (default: exp)

## Making Predictions

After training, you can use the `predict_yolo.py` script to make predictions:

```bash
python predict_yolo.py --model runs/detect/exp/weights/best.pt --source datasets/test/images --conf 0.25
```

### Prediction Arguments

- `--model`: Path to trained model weights (required)
- `--source`: Path to image or directory of images for prediction (required)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--save-dir`: Directory to save results (default: predictions)

## Results

Training results will be saved in the specified project directory (default: `runs/detect/exp`). This includes:
- Model weights (best.pt and last.pt)
- Training metrics and plots
- Validation results

Prediction results will be saved in the specified save directory (default: `predictions/exp`).
