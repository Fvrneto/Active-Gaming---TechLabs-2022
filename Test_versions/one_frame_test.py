#Import modules

import torch
import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from time import time
from PIL import Image

Path_model = 'C:/Users/Francisco/Desktop/Techlabs 2022.01/Active-Gaming-TechLabs-2022-main/complete_model.pt'
 
#model = torch.load(Path_model)
model = torch.load(Path_model, map_location=torch.device('cpu'))
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Funtions

def frame_builder(img_path, transform):
        image = Image.open(img_path)
        image = image.convert("RGB")
        image = transform(image)
        return (image)

#######################1 picture test

frame_path = 'C:/Users/Francisco/Desktop/Techlabs 2022.01/Active-Gaming-TechLabs-2022-main/Arm positions images/up_10.jpg'
#frame_path = 'C:/Users/franc/Desktop/Techlabs-2022.1/Github/Active-Gaming-TechLabs-2022/Arm positions images/down_36.jpg'

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((256, 256))
    ])

frame = frame_builder(frame_path, transform = transform)
frame = torch.unsqueeze(frame,0)
#print(frame.shape)

#predict 

output = model(frame)

prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)

if (prediction == 0):
    print ('down')
if (prediction == 1):
    print ('left')
if (prediction == 2):
    print ('right')
if (prediction == 3):
    print ('up')

###################1 frame test



