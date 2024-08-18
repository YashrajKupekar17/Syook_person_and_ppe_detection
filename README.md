# PPE Detection Using YOLOv8

## Project Overview
This project implements a object detection system using YOLOv8 to identify persons and personal protective equipment (PPE) in various environments. The dataset consists of images and annotations for classes such as hard hats, gloves, masks, glasses, boots, vests, PPE suits, ear protectors, and safety harnesses. The goal is to train two models: one for detecting persons in whole images and another for detecting PPE in cropped images of persons.

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
git clone https://github.com/YashrajKupekar17/Syook_person_and_ppe_detection
cd Syook_person_and_ppe_detection
```
### Step 2: Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```
### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```
## Usage
### Step 1: Prepare the Dataset
Download the dataset from the provided [link](https://drive.google.com/file/d/1myGjrJZSWPT6LYOshF9gfikyXaTCBUWb/view?usp=sharing). Extract the contents of Datasets.zip, which contains the images and annotations directories along with classes.txt.
### Step 2: Convert Annotations
Run the following command to convert annotations from PascalVOC format to YOLOv8 format:
```bash
python pascalVOC_to_yolo.py --voc_dir path/to/annotations --yolo_dir path/to/yolov8_annotations
```
### Step 3: Train the Models
#### Train Person Detection Model
To train the YOLOv8 model for person detection, run:
```bash 
python train.py --data data.yaml --weights yolov8.pt --cfg yolov8.yaml --epochs 50
```
#### Train PPE Detection Model
To train the YOLOv8 model for PPE detection on cropped images, ensure you have implemented the logic to crop images based on the detected persons. Then run:
``` bash 
python train.py --data ppe_data.yaml --weights yolov8.pt --cfg yolov8.yaml --epochs 50
```
### Step 4: Run Inference
To perform inference using both models, run:
```bash 
python inference.py --input_dir path/to/images --output_dir path/to/output --person_det_model path/to/person_weights.pt --ppe_detection_model path/to/ppe_weights.pt
```


## Evaluation Metrics
The models will be evaluated using metrics such as:

## Precision
Recall

F1 Score

Mean Average Precision (mAP)

## Report
A report containing the approaches, learning outcomes, and evaluation metrics is provided in PDF format. It includes:

Logic used for model training.

Challenges faced and solutions implemented.

[Demonstration Video](https://www.loom.com/share/da309a6333944401a8c51071701ac4fa?sid=e842ba3a-aae8-4a23-bd9b-2f555864d4c6)
[Project Report](https://docs.google.com/document/d/15E0lkguBZdbB3cAMEWSzF0p5pZrVxCtL3LASgPivuCs/edit?usp=sharing)
