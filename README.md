# Active-Gaming-TechLabs-2022

This repository is the 2022 TechLabs AI project called “Active Gaming”. The goal of the project was to train an AI to enable controlling a video game by movements detected via WebCam. More specifically, it was planned to control the popular game “Snake” by lifting the arms up, down, left, or right. We have achieved to train an AI to take a snapshot with the WebCam and predict the direction the person is showing. Several models have been tested, best results in the prediction of a WebCam image were achieved with the model based on the pretrained Shufflenet_v2_x1_0 model. 


*Note: The images used for training have been removed from the public repository due to privacy reasons*


The main files in the repository are :

**GoogLeNet_model_trainer.ipynb**: Google Colab code for training the model based on the GoogLeNet pretrained model

**Other_models_trainer.ipynb***: Code for training several models based on different pretrained models

**Predicting_one_frame.ipynb**: Code applying the trained models. It consists of two parts: First, test application to one image from the dataset. Second, opening the WebCam, taking a snapshot and predicting a position based on the trained model selected. The program should be run completely and the position acquired. Once the snapshot is taken, it needs to be opened and confirmed by typing “y” in the keyboard. Then, a prediction will be made as to which position the person is showing.

**parameters_full.csv**: Dictionary assigning IDs to the images: 0 = down, 1 = left, 2 = right, 3 = up 


The repository also contains the following folders:

1.	**Test versions**

    The folder contains preliminary versions, tests of model training and opening images. The older versions have kept in the repository to look back at the process    throughout the project phase. *Note: as the files have been shifted into a different folder, they might not run without adjustments to the file paths anymore*

2.	**Trained models**

    Contains several trained models based on different pretrained models:

      GoogLeNet

      Resnet18

      Mobilenet_v2

      Shufflenet_v2_x1_0




The models were trained using Google Colab. The firs model was based on the GoogLeNet pretrained model. The code can be found in the following Google Drive:
https://drive.google.com/drive/folders/1vSbKkymLKtc7z5EK7DNZUxq6nO2TwSKx


Python modules needed to run the programs:

torch

torchvision

numpy

cv2

time

PIL

matplotlib

os
