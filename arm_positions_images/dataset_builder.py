"""
@author: Francisco
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

#writing builder

class data_frame_builder(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.parameters = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.parameters) #number of images
    
    def __getitem__(self, index):
        img_path = os.patch.join(self.root_dir, self.parameters.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.parameters.iloc[index, 1]))
        
        if self.transform:
            image = self.transform(image)
            
        return (image, y_label)
    

