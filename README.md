# E6692 Spring 2025: Final Project - Autonomous navigation for blind on Jetson Nano

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

## How to modify this README.md file
Students need to maintain this repo to have a "professional look":
* Remove the instructions (this text)
* Provide description of the topic/project
* Provide organization of this repo 
* Add all relevant links: name of Google docs and link to them, links to public repos used, etc.
* For paper reviews it should include the organization of the directory, brief description how to run the code, what is in the code, links to relevant papers, links to relevant githubs, etc...

## INSTRUCTIONS for (re) naming the student's repository for the final project with one student:
* Students need to use the following naming rules for the repository with their solutions: e6692-2025Spring-FinalProject-GroupID-UNI 
(the first part "e6692-2025Spring-FinalProject" will probably be inherited from the assignment, so only UNI needs to be added) 
* Initially, the system may give the repo a name which ends with a student's Github userid. 
The student must change that name and replace it with the name requested in the point above (containing their UNI)
* Good Example: e6692-2025Spring-FinalProject-GroupID-zz9999;   Bad example: e6692-2025Spring-FinalProject-ee6040-2025Spring-FinalProject-GroupID-zz9999.
* This change can be done from the "Settings" tab which is located on the repo page.

## INSTRUCTIONS for naming the students' repository for the final project with more students. 
(Students need to use a 4-letter groupID/teamID): 
* Template: e6692-2025Spring-FinalProject-GroupID-UNI1-UNI2-UNI3. -> Example: e6692-2025Spring-FinalProject-MEME-zz9999-aa9999-aa0000.
