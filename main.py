import numpy as np
import torchvision
from torchvision import transforms
import re
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from sklearn.model_selection import KFold
from torch import Tensor,optim
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import recall_score
import random
from torchvision.transforms import Normalize
from Dataloader.Dataloader import MonkeyPoxDataLoader
import warnings
import argparse
warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
import time

def main(args):
    params = {'batch_size':args.batch,
            'shuffle':True}
    rootdir = '/home/sci/mkaranam/Desktop/CS6190---ProbabilisticMLProject/data/'
    img_dir = "/home/sci/mkaranam/Desktop/CS6190---ProbabilisticMLProject/OriginalImages/OriginalImages/Total_Data/"
    scratchDir = './Results'
    
    tr_csv_file = os.path.join(rootdir,"trainMonkeypox.csv")
    cv_csv_file = os.path.join(rootdir,"cvMonkeypox.csv")
    te_csv_file = os.path.join(rootdir,"testMonkeypox.csv")

    trans = transforms.Compose([transforms.ToTensor(),Normalize(mean=(0.485), std=(0.229))])
    test_trans = transforms.Compose([transforms.ToTensor(),Normalize(mean=(0.485), std=(0.229))])
    import pdb;pdb.set_trace()
    train = MonkeyPoxDataLoader(tr_csv_file, img_dir, transform=trans)
    train_dataloader = torch.utils.data.DataLoader(train, **params)
    train_dataloader_eval = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    cv = MonkeyPoxDataLoader(cv_csv_file, img_dir, transform=test_trans)

    cv_dataloader = torch.utils.data.DataLoader(cv, **params)
    cv_dataloader_eval = torch.utils.data.DataLoader(cv,batch_size=1, shuffle=True)

    test = MonkeyPoxDataLoader(te_csv_file, img_dir, transform=test_trans)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Short sample app')
        parser.add_argument('-batch'             ,type=int  , action="store", dest='batch'       , default=10       )
        args = parser.parse_args()
        main(args)