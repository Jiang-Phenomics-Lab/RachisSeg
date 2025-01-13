# RachisSeg: Automated Rachis Phenotyping Pipeline

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-1.12.0-brightgreen.svg)
![Detectron2 Version](https://img.shields.io/badge/detectron2-0.6-brightgreen.svg)

## Introduction
- RachisSeg is a high-throughput, automated phenotyping pipeline designed to segment rachis internodes and quantitatively analyze rachis traits in wheat.
- By integrating deep learning networks with traditional image analysis algorithms, RachisSeg offers precise measurements, reducing the labor and errors associated with manual phenotyping.
- RachisSeg based on [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://ieeexplore.ieee.org/document/7485869)

 
## Requirements:  
```angular2
torch==1.12.0
detectron2==0.6
numpy==1.26.0
opencv-python==4.8.1.78
scikit-image==0.19.2
skfmm==1.5.3
matplotlib==3.5.1
scipy==1.10.1
```
## Installation

### Clone the Repository: 
```
git clone https://github.com/Jiang-Phenomics-Lab/RachisSeg.git
cd RachisSeg
```

### Create and Activate a Virtual Environment (Optional but Recommended): 
```
python -m venv venv
source venv/bin/activate  # For Windows users: `venv\Scripts\activate`
```

### Install Dependencies: 
```
pip install -r requirements.txt
```

## Code Structure
All codes were written by python.

**coco_merge.py** : Merge of multiple coco formated json file.

**cropRachis.py** : Crop individual rachis from original scaned images.

**evaluator.py** : evaluation of trained model.

**rachis_prediction.py** : Prediction of rachis node and extraction of rachis phenotype.

**rachis_train.py** : Train of rachis node objective detection.

**register_dataset.py** : Spilit rachis dataset and register the dataset.

## Quickly Start

Follow this quick start guide to get RachisSeg up and running:

### 1)Prepare the Rachis Image Dataset: 

Collect and organize rachis image data, ensuring high image quality and accurate annotations.

Use register_dataset.py to split and register the dataset.

```
python register_dataset.py
```

### 2) Training RachisSeg

Run the training script:

```
python rachis_train.py
```

### 3) Testing RachisSeg

Use the trained model to make predictions by running rachis_prediction.py:
```
python rachis_prediction.py
```

## examples

<img src="https://github.com/Jiang-Phenomics-Lab/RachisSeg/dataset/rachis_phenotyping_pipeline.png" width="6496">

**Contact**
For questions or support, please contact: rxlu@genetics.ac.cn
