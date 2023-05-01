from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
# from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import cv2
import torchvision.transforms as transforms 
import pickle as pk
from PIL import Image
import random



class MonkeyPoxDataLoader(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.monkey_pox = pd.read_csv(csv_file)
        self.root_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.monkey_pox)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.monkey_pox.iloc[idx, 0])
        image = np.array(Image.open(img_name))
        
        label = self.monkey_pox.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

class MonkeyPoxRandAugDataLoader(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.monkey_pox = pd.read_csv(csv_file)
        self.root_dir = img_dir
        self.transform = transform
        self.rgb_transform = transforms.Compose(
            [ transforms.Lambda(lambda x: x.repeat(3, 1, 1) )]
        )
 

    def __len__(self):
        return len(self.monkey_pox)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.monkey_pox.iloc[idx, 0])
        image = Image.open(img_name)
        
        label = self.monkey_pox.iloc[idx, 1]
        if len(image.mode) == 1:
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
            # image = (image - image.min()) / (image.max() - image.min())
        return image, label
