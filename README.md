# Active-Gaming-TechLabs-2022

This repository is the 2022 TechLabs AI project called “Active Gaming”. The goal of the project was to train an AI to enable controlling a video game by movements detected via WebCam. More specifically, it was planned to control the popular game “Snake” by lifting the arms up, down, left, or right. We have achieved to train an AI to take a snapshot with the WebCam and predict the direction the person is showing. 

<<<<<<< HEAD
=======

>>>>>>> af73dde2a60a173998b55055d66717e504467bc9
The main files in the repository are :

**GoogLeNet_model_trainer.ipynb**: Google Colab code for training the model based on the GoogLeNet pretrained model

<<<<<<< HEAD
**Other_models_trainer.ipynb***: Code for training several models based on different pretrained models
=======
**Other_models_trainer.ipynb**: Code for training several models based on different pretrained models
>>>>>>> af73dde2a60a173998b55055d66717e504467bc9

**Predicting_one_frame.ipynb**: Code applying the trained models. It consists of two parts: First, test application to one image from the dataset. Second, opening the WebCam, taking a snapshot and predicting a position based on the trained model selected. The program should be run completely and the position acquired. Once the snapshot is taken, it needs to be opened and confirmed by typing “y” in the keyboard. Then, a prediction will be made as to which position the person is showing.

**parameters_full.csv**: Dictionary assigning IDs to the images: 0 = down, 1 = left, 2 = right, 3 = up 


The repository also contains the following folders:

1. **Arm positions images**

  	Dataset created for training the models. Contains 64 images for “down”, 63 images for “left”, 66 images for “right” and 62 images for “up”. Pointing up and down is done with both hands, left and right with only one hand. The images where the left hand was raised by the person are labeled as “left” and if the right hand was raised it is labeled as “right”.
<<<<<<< HEAD

2.	**Test versions**

    The folder contains preliminary versions, tests of model training and opening images. The older versions have kept in the repository to look back at the process    throughout the project phase.

3.	**Trained models**

    Contains several trained models based on different pretrained models:

      GoogLeNet

      Resnet18

      Mobilenet_v2

      Shufflenet_v2_x1_0



=======
2.	**Test versions**

    The folder contains preliminary versions, tests of model training and opening images. The older versions have kept in the repository to look back at the process    throughout the project phase.
3.	**Trained models**

    Contains several trained models based on different pretrained models:
      GoogLeNet,
      Resnet18,
      Mobilenet_v2,
      Shufflenet_v2_x1_0


>>>>>>> af73dde2a60a173998b55055d66717e504467bc9
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
