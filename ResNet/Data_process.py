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

data_dir = 