# RachisSeg: Automated Rachis Phenotyping Pipeline



## Introduction
 RachisSeg is a high-throughput, automated phenotyping pipeline designed to segment rachis internodes and quantitatively analyze rachis traits in wheat. By integrating deep learning networks with traditional image analysis algorithms, RachisSeg offers precise measurements, reducing the labor and errors associated with manual phenotyping.

 
### Requirements:  
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
### Code Structure
All codes were written by python.

**coco_merge.py** : Merge of multiple coco formated json file.

**cropRachis.py** : Crop individual rachis from original scaned images.

**evaluator.py** : evaluation of trained model.

**rachis_prediction.py** : Prediction of rachis node and extraction of rachis phenotype.

**rachis_train.py** : Train of rachis node objective detection.

**register_dataset.py** : Spilit rachis dataset and register the dataset.

## Quickly Start
### 1)Rachis image dataset: 
1. 

### 2) Training RachisSeg
1. 
---   
### 3) Testing RachisSeg
1.

## examples
 
**Contact**
For questions or support, please contact: rxlu@genetics.ac.cn
