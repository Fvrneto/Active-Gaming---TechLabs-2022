#Import modules

import cv2
import torch
import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from time import time
from PIL import Image

Path = 'C:/Users/franc/Desktop/Techlabs-2022.1/Github/Active-Gaming-TechLabs-2022/complete_model.pt'

model = torch.load(Path)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Funtions

def frame_builder(img_path, transform):
        image = Image.open(img_path)
        image = image.convert("RGB")
        image = transform(image)
        return (image)

def score_frame(frame, model):
    
    model.to(device)
    results = model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    cord = results.xyxyn[0][:, :-1].numpy()
    
    return labels, cord

def plot_boxes(self, results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.2: 
            continue
        x1 = int(row[0]*x_shape)
        y1 = int(row[1]*y_shape)
        x2 = int(row[2]*x_shape)
        y2 = int(row[3]*y_shape)
        bgr = (0, 255, 0) # color of the box
        classes = self.model.names # Get the name of label index
        label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
        cv2.rectangle(frame, \
                      (x1, y1), (x2, y2), \
                       bgr, 2) #Plot the boxes
        cv2.putText(frame,\
                    classes[labels[i]], \
                    (x1, y1), \
                    label_font, 0.9, bgr, 2) #Put a label over box.
        return frame

#1 frame test

frame_path = 'C:/Users/franc/Desktop/Techlabs-2022.1/Github/Active-Gaming-TechLabs-2022/Arm positions images/down_36.jpg'

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((256, 256))
    ])

frame = frame_builder(frame_path, transform = transform)

print(frame.shape)

results = model(frame)

results = score_frame(frame, model)
    
