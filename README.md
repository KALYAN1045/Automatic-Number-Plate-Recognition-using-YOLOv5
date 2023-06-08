
# Automatic Number Plate Detection 🚘

[![Screenshot-2023-05-28-001749.png](https://i.postimg.cc/rFTXByg8/Screenshot-2023-05-28-001749.png)](https://postimg.cc/Jtp2JL9g)

The goal of this project was to create a model that can accurately detect number plates on cars and bikes. This model can be used along with the video data generated from CCTV cameras that are installed on highways and roadways to detect number plate of vechicles that commit traffic violations.


## ℹ️ Model Information

Object Detection is carried out in this project to detect Number Plates on vehicles. YOLOv5s model was used along with transfer learning for training and testing. Model accurately generates bounding boxes around Number Plates

More Info on YOLOv5: https://github.com/ultralytics/yolov5

## 📚 Dataset

The dataset I used for training for Number plate detection:  

* https://www.kaggle.com/datasets/elysian01/car-number-plate-detection

## ✍️ Installation

install requirements.txt in a Python>=3.7.0 environment.

    git clone https://github.com/KALYAN1045/Automatic-Number-Plate-Recognition-using-YOLOv5.git  # clone
    
    cd Automatic-Number-Plate-Recognition-using-YOLOv5
    
    pip install -r requirements.txt  # install
## 🔐 Implementation

Object Detection pipeline cinsists of 3 parts:
**Training, Validation and Testing**

### Training:

YOLOv5s was trained on Google Colab with following hyperparameters:

* Input Image Size: 640
* Batch Size: 16
* Epochs: 300 (converged in 198 epochs)
* Pretrained Weights: yolov5s.pt

The training dataset consisted of 400 images along with their class and bounding box details mentioned in a txt file in yolo format. The training was set to run for 300 epochs but the model converged in 198 epochs and the training was stopped.

### Validation:

The validation dataset consisted of 100 images along with their class and bounding box details in a txt file in yolo format for validation purpose. In validation phase the model reached Mean Average Precision (mAP) of 0.91

Following are some of the image results generated in validation phase:

![alt text](https://github.com/wasdac9/automatic-number-plate-recognition/blob/main/val_pred.jpg?raw=true)

At the end of training and validation epochs, a weights file ("best.pt") is generated which consists of all the learned parameters of the model. 

Refer to "ANPD_OD.ipynb" for more info about the training and validation process.

### Testing Phase
The model was tested on various images and videos and the model generated accurate class and bounding box predictions. The weights file called "/weights/best.pt" that was generate in the training phase was used for inference in testing phase. Testing was carried out in PyTorch, and OpenCV was used when working with images and videos. OpenCV helped in loading, saving, drawing bounding boxes, and displaying text regarding class name and class confidence values.
## 📦 Result

[![result.png](https://i.postimg.cc/G2tR6KCw/result.png)](https://postimg.cc/3yszvgNn)
## 📖 Summary
An accurate object detection model was created to carry out Automatic Number Plate Recognition using YOLOv5 and transfer learning along with Pytorch. The accuracy of bounding boxes and the frame rate was found to be good. 


## 🛑 Limitations
*  Even though the accuracy and frame rate was good, the model sometimes wrongly detected various signs on road as number plate. A bigger dataset is required to improve upon the current model accuracy.
* The model in its current state works very well on CUDA devices getting around 32 FPS but does perform well on CPU devices. Hence GPU hardware is required for smooth frame rate outputs.
* It can't detect the number plates of bikes due to various factors and it also has a problem with the camera quality.

## 🚀 Future works
* A bigger dataset can be used to train the model for more number of epochs to reduce the false positive predictions.
* This detection model can be uploaded on edge devices connected to CCTV cameras to carry out Number Plate Recognition live on the road.
* CCTV video footage can be used to read number plate of vehicles that commit traffic violations.
* The bounding box around the license plate can be cropped and Optical Character Recognition can be used to actually read the number plate.
## ♨️ Tech Stack

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)







