# PPE Detection Using YOLOv8

## Project Overview
This project implements a dual-object detection system using YOLOv8 to identify persons and personal protective equipment (PPE) in various environments. The dataset consists of images and annotations for classes such as hard hats, gloves, masks, glasses, boots, vests, PPE suits, ear protectors, and safety harnesses. The goal is to train two models: one for detecting persons in whole images and another for detecting PPE in cropped images of persons.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
  - [Annotation Conversion](#annotation-conversion)
  - [Model Training](#model-training)
  - [Inference](#inference)
- [Evaluation Metrics](#evaluation-metrics)
- [Report](#report)
- [Demonstration Video](#demonstration-video)
- [License](#license)

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/ppe-detection.git
cd ppe-detection
Step 2: Set Up a Virtual Environment
bash

Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Step 3: Install Required Packages
bash
Copy code
pip install -r requirements.txt
Usage

Step 1: Prepare the Dataset
Download the dataset from the provided link. Extract the contents of Datasets.zip, which contains the images and annotations directories along with classes.txt.

Step 2: Convert Annotations
Run the following command to convert annotations from PascalVOC format to YOLOv8 format:

bash
Copy code
python pascalVOC_to_yolo.py --voc_dir path/to/annotations --yolo_dir path/to/yolov8_annotations
Step 3: Train the Models
Train Person Detection Model

To train the YOLOv8 model for person detection, run:

bash
Copy code
python train.py --data data.yaml --weights yolov8.pt --cfg yolov8.yaml --epochs 50
Train PPE Detection Model

To train the YOLOv8 model for PPE detection on cropped images, ensure you have implemented the logic to crop images based on the detected persons. Then run:

bash
Copy code
python train.py --data ppe_data.yaml --weights yolov8.pt --cfg yolov8.yaml --epochs 50
Step 4: Run Inference
To perform inference using both models, run:

bash
Copy code
python inference.py --input_dir path/to/images --output_dir path/to/output --person_det_model path/to/person_weights.pt --ppe_detection_model path/to/ppe_weights.pt
Scripts

Annotation Conversion
Script: pascalVOC_to_yolo.py
Description: Converts annotations from PascalVOC format to YOLOv8 format.
Arguments:
--voc_dir: Path to the directory containing PascalVOC annotations.
--yolo_dir: Path to the directory where YOLOv8 annotations will be saved.
Model Training
Script: train.py
Description: Trains YOLOv8 models for person detection and PPE detection.
Inference
Script: inference.py
Description: Performs inference on input images using both models and saves the results.
Arguments:
--input_dir: Path to the directory containing input images.
--output_dir: Path to the directory where inference results will be saved.
--person_det_model: Path to the trained person detection model weights.
--ppe_detection_model: Path to the trained PPE detection model weights.
Evaluation Metrics

The models will be evaluated using metrics such as:

Precision
Recall
F1 Score
Mean Average Precision (mAP)
Report

A report containing the approaches, learning outcomes, and evaluation metrics is provided in PDF format. It includes:

Logic used for model training.
Challenges faced and solutions implemented.
Demonstration Video

A demonstration of the project can be viewed here.
