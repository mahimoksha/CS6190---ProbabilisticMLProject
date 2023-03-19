import time
import numpy as np
from tqdm import tqdm
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
from torch import Tensor, optim
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import recall_score
import random
from torchvision.transforms import Normalize
from torchvision import models
from Dataloader.Dataloader import MonkeyPoxDataLoader,MonkeyPoxRandAugDataLoader
import warnings
import argparse
from torchvision.transforms.autoaugment import AutoAugmentPolicy
warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_model(name, num_classes=1, fixed_pretrain=True):
    if name == "resnet50":
        model_ft = models.resnet50(pretrained=True)
    elif name == "resnet34":
        model_ft = models.resnet34(pretrained=True)
    else:
        raise NotImplementedError(
            f"Invalid model name: {name}. Please provide one of the following model names: \n 1. resnet50 \n 2. resnet34")

    if fixed_pretrain:
        for param in model_ft.parameters():
            param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model = nn.Sequential(
        model_ft,
        torch.nn.Sigmoid()
    )
    model = model.to(device)
    print(model)
    return model


def one_epoch(model, loader, opt, loss_function, train=True):
    torch.cuda.empty_cache()
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    for input, output in tqdm(loader):
        opt.zero_grad()
        input = input.to(device)
        output = torch.FloatTensor([[o.to(device)] for o in output]).to(device)
        pred = model(input)
        # print(input.shape, output, pred, output.shape, pred.shape)
        loss = loss_function(pred, output)
        if train:
            loss.backward()
            opt.step()
        losses.append(loss.item())
    return np.mean(losses), model


def trainer(args,model_name, epochs, train_loader, val_loader, training_logs, scratchDir):
    model = get_model(model_name)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    print(f"Epoch\ttrain_loss\tval_loss")
    training_logs.write(f"Epoch\ttrain_loss\tval_loss\n")
    best_val_loss = 10000
    for i in range(epochs):
        train_loss, model = one_epoch(
            model, train_loader, optimizer, loss_function)
        val_loss, model = one_epoch(
            model, val_loader, optimizer, loss_function, train=False)
        print(f"{i+1}\t{train_loss}\t{val_loss}")
        training_logs.write(f"{i+1}\t{train_loss}\t{val_loss}\n")
        if val_loss < best_val_loss:
            torch.save(model.state_dict(),
                       f'{scratchDir}/{model_name}_best_model.torch')


def eval(model_name, loader, training_logs, scratchDir):
    model = get_model(model_name)
    model.load_state_dict(torch.load(
        f'{scratchDir}/{model_name}_best_model.torch'))
    model.eval()
    training_logs.write(f"\nTest predictions. \n")
    training_logs.write(f"\npred\ttrue\n")
    correct = 0
    for input, output in loader:
        input = input.to(device)
        output = output.to(device)
        pred = model(input)
        training_logs.write(
            f"{pred[-1].squeeze().detach().cpu().numpy()}\t{output[-1].squeeze().detach().cpu().numpy()}\n")
        prediction = 0 if pred[-1].squeeze().detach().cpu().numpy() < 0.5 else 1
        if prediction == output[-1]:
            correct += 1

    print("Done")
    print(f"Accuracy: {correct/len(loader)}")
    training_logs.write(f"\nAccuracy: {correct/len(loader)}\n")


def main(args):
        params = {'batch_size': args.batch,
                'shuffle': True}
        rootdir = '/usr/sci/scratch/Moksha/CS6190_project/'
        csv_dir = os.path.join(rootdir, "data")
        img_dir = "/usr/sci/scratch/Moksha/CS6190_project/OriginalImages/OriginalImages/Total_Data/"
        scratchDir = './Results'

        tr_csv_file = os.path.join(csv_dir, "trainMonkeypox.csv")
        cv_csv_file = os.path.join(csv_dir, "cvMonkeypox.csv")
        te_csv_file = os.path.join(csv_dir, "testMonkeypox.csv")
        if args.runningType =="noAug":
                trans = transforms.Compose(
                [transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])
                test_trans = transforms.Compose(
                [transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])
                train = MonkeyPoxDataLoader(tr_csv_file, img_dir, transform=trans)
                train_dataloader = torch.utils.data.DataLoader(train, **params)
                train_dataloader_eval = torch.utils.data.DataLoader(
                train, batch_size=1, shuffle=True)
                cv = MonkeyPoxDataLoader(cv_csv_file, img_dir, transform=test_trans)

                cv_dataloader = torch.utils.data.DataLoader(cv, **params)
                cv_dataloader_eval = torch.utils.data.DataLoader(
                cv, batch_size=1, shuffle=True)

                test = MonkeyPoxDataLoader(te_csv_file, img_dir, transform=test_trans)
                test_dataloader = torch.utils.data.DataLoader(
                test, batch_size=1, shuffle=True)
        elif args.runningType =="RandAug":
                trans = transforms.Compose(
                [transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])
                test_trans = transforms.Compose(
                [transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])
                train = MonkeyPoxRandAugDataLoader(tr_csv_file, img_dir, transform=trans)
                train_dataloader = torch.utils.data.DataLoader(train, **params)
                train_dataloader_eval = torch.utils.data.DataLoader(
                train, batch_size=1, shuffle=True)
                cv = MonkeyPoxRandAugDataLoader(cv_csv_file, img_dir, transform=test_trans)

                cv_dataloader = torch.utils.data.DataLoader(cv, **params)
                cv_dataloader_eval = torch.utils.data.DataLoader(
                cv, batch_size=1, shuffle=True)

                test = MonkeyPoxRandAugDataLoader(te_csv_file, img_dir, transform=test_trans)
                test_dataloader = torch.utils.data.DataLoader(
                test, batch_size=1, shuffle=True)


        # model_name = "resnet50"
        model_name = args.model_name
        training_logs = open(f"{scratchDir}/training_log_{model_name}.txt", 'w')
        epochs = args.epochs
        trainer(args,model_name, epochs, train_dataloader,
                cv_dataloader, training_logs, scratchDir)
        eval(model_name, test_dataloader, training_logs, scratchDir)
        training_logs.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-batch', type=int, action="store",
                        dest='batch', default=10)
    parser.add_argument('-type'              ,type=str  , action="store", dest='runningType'   , default='noAug')
    parser.add_argument('-model_name'              ,type=str  , action="store", dest='model_name'   , default='resnet50')
    parser.add_argument('-epochs'            ,type=int  , action="store", dest='epochs', default=100       )
    parser.add_argument('-lr'           ,type=float  , action="store", dest='lr'           , default=0.001     )
    args = parser.parse_args()
    main(args)
