import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

data_dir = "CNN/DeepFake/RelGAN"
categories = ['real','fake']

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size

# print(train_size)
# print(test_size)

train_dataset, test_dastset = random_split(dataset,[train_size,test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dastset,batch_size=32,shuffle=False)