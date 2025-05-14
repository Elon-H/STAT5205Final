# STAT5205 Spring 2025: Final Project - Autonomous navigation for blind on Jetson Nano

## Project Description
This project implements a real-time obstacle detection system using YOLOv5 for autonomous navigation. The system is optimized for deployment on Jetson Nano, utilizing TensorRT for efficient inference. The model is trained to detect 3 classes: "walkable_area", "human" and "obstacle", providing crucial information for autonomous navigation systems.

## Repository Organization
```
├── dataset/                 # Dataset directory
│   ├── images/             # Image files
│   │   ├── train/         # Training images
│   │   └── val/           # Validation images
│   ├── labels/             # Label files
│   │   ├── train/         # Training labels
│   │   └── val/           # Validation labels
│   └── data.yaml          # Dataset configuration
├── segment/                # Segmentation model code
│   ├── predict.py         # Inference script
│   └── val.py            # Validation script
├── best.engine            # TensorRT optimized model
└── best.pt               # Original PyTorch model
```

## Model Details
- Architecture: YOLOv5 with segmentation head
- Classes: walkable_area, human, obstacle
- Input size: 640x640
- Optimized for: Jetson Nano with TensorRT
- Precision: FP16 (half-precision)

## Performance
- Real-time inference on Jetson Nano
- Optimized for edge deployment
- Support for both TensorRT and PyTorch models

## References
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [DeepWay.v2: Autonomous Navigation for Blind People](https://github.com/satinder147/DeepWay.v2)
- [SightWalk: Open-Source Sidewalk Navigation Software for Visually-Impaired Individuals using Multithreaded CNNs](https://github.com/team8/outdoor-blind-navigation)
