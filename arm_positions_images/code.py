# -*- coding: utf-8 -*-
"""
@author: Francisco
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# usder modules

from dataset_builder import data_frame_builder

# Load data

dataset = data_frame_builder(csv_file = 'parameters.csv', root_dir = 'builder_test', transform = transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [2,4])

train_loader = DataLoader(dataset=train_set, shuffle=True)
test_loader = DataLoader(dataset=test_set, shuffle=True)
