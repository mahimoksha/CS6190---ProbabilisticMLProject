import numpy as np
from tqdm import tqdm
from torchvision import transforms
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.transforms import Normalize
from torchvision import models
from Dataloader.Dataloader import MonkeyPoxRandAugDataLoader
import warnings
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.transforms.autoaugment import AutoAugmentPolicy
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def trainer(args,model_name, train_loader, val_loader, training_logs, scratchDir):
    model = get_model(model_name)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    print(f"Epoch\ttrain_loss\tval_loss")
    training_logs.write(f"Epoch\ttrain_loss\tval_loss\n")
    best_val_loss = 10000
    train_loss_arr = []
    val_loss_arr = []
    for i in range(args.epochs):
        train_loss, model = one_epoch(
            model, train_loader, optimizer, loss_function)
        val_loss, model = one_epoch(
            model, val_loader, optimizer, loss_function, train=False)
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        print(f"{i+1}\t{train_loss}\t{val_loss}")
        training_logs.write(f"{i+1}\t{train_loss}\t{val_loss}\n")
        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(),
                       f'{scratchDir}/{model_name}_{args.runningType}_best_model.torch')
            best_val_loss = val_loss

    return train_loss_arr, val_loss_arr

def eval(args, model_name, loader, training_logs, scratchDir):
    model = get_model(model_name)
    model.load_state_dict(torch.load(
        f'{scratchDir}/{model_name}_{args.runningType}_best_model.torch'))
    model.eval()
    training_logs.write(f"\nTest predictions. \n")
    training_logs.write(f"\npred\ttrue\n")
    correct = 0
    correct_cm = []
    pred_cm = []
    for input, output in loader:
        input = input.to(device)
        output = output.to(device)
        pred = model(input)
        training_logs.write(
            f"{pred[-1].squeeze().detach().cpu().numpy()}\t{output[-1].squeeze().detach().cpu().numpy()}\n")
        prediction = 0 if pred[-1].squeeze().detach().cpu().numpy() < 0.5 else 1
        if prediction == output[-1]:
            correct += 1
        correct_cm.append(output[-1].item())
        pred_cm.append(prediction)

    print("Done")
    print(f"Accuracy: {correct/len(loader)}")
    training_logs.write(f"\nAccuracy: {correct/len(loader)}\n")
    return correct_cm, pred_cm


def main(args):
    params = {'batch_size': args.batch,
            'shuffle': True}
    rootdir = '/usr/sci/scratch/Moksha/CS6190_project/'
    csv_dir = os.path.join(rootdir, "data")
    img_dir =  "/usr/sci/scratch/Moksha/CS6190_project/OriginalImages/OriginalImages/original+generated/" # "/usr/sci/scratch/Moksha/CS6190_project/OriginalImages/OriginalImages/Total_Data/"
    scratchDir = './Results'

    tr_csv_file = os.path.join(csv_dir, "vae_trainMonkeypox.csv")
    cv_csv_file = os.path.join(csv_dir, "vae_cvMonkeypox.csv")
    te_csv_file = os.path.join(csv_dir, "testMonkeypox.csv")
    if args.runningType =="noAug" or "vae" in args.runningType:
        trans = transforms.Compose(
        [transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])
        test_trans = transforms.Compose(
        [transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])
            
    elif args.runningType =="PretrainedAug":
        trans = transforms.Compose(
        [transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])
        test_trans = transforms.Compose(
        [transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])

    elif args.runningType =="RandAug":
        trans = transforms.Compose(
        [transforms.RandAugment(num_ops=4, magnitude=14),transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])
        test_trans = transforms.Compose(
        [transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])

    elif args.runningType =="GenericAug":
        trans = transforms.Compose([
            transforms.ToTensor(), 
            Normalize(mean=(0.485), std=(0.229)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((200, 200)),
            transforms.RandomRotation(30),
            transforms.RandomAutocontrast(p=0.5)
        ])
        test_trans = transforms.Compose(
        [transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])
    
    elif args.runningType == "EnsembleAug":
        trans = transforms.Compose([
            transforms.RandAugment(num_ops=4, magnitude=14),
            transforms.ToTensor(), 
            Normalize(mean=(0.485), std=(0.229)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((200, 200)),
            transforms.RandomRotation(30),
            transforms.RandomAutocontrast(p=0.5)
        ])
        test_trans = transforms.Compose(
        [transforms.ToTensor(), Normalize(mean=(0.485), std=(0.229))])
    else:
        raise NotImplementedError(f"Invalid augmentation type: {args.runningType}")

    train = MonkeyPoxRandAugDataLoader(tr_csv_file, img_dir, transform=trans)
    train_dataloader = torch.utils.data.DataLoader(train, **params)
    cv = MonkeyPoxRandAugDataLoader(cv_csv_file, img_dir, transform=test_trans)
    cv_dataloader = torch.utils.data.DataLoader(cv, **params)

    test = MonkeyPoxRandAugDataLoader(te_csv_file, img_dir, transform=test_trans)
    test_dataloader = torch.utils.data.DataLoader(
    test, batch_size=1, shuffle=True)

    model_name = args.model_name
    training_logs = open(f"{scratchDir}/training_log_{model_name}_{args.runningType}.txt", 'w')
    train_loss_arr, val_loss_arr = trainer(args,model_name, train_dataloader,
            cv_dataloader, training_logs, scratchDir)
    fig, ax = plt.subplots()
    fig.set_size_inches(20,16)
    ax.plot(train_loss_arr, label="train")
    ax.plot(val_loss_arr, label="validation")
    ax.legend()
    fig.savefig(os.path.join(scratchDir,"train_val_loss_"+model_name+"_"+args.runningType+".png"),transparent=True,bbox_inches='tight') 
    orig, pred = eval(args, model_name, test_dataloader, training_logs, scratchDir)
    cm = confusion_matrix(np.array(orig), np.array(pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(scratchDir,"confusion_matrix_"+model_name+"_"+args.runningType+".png"),transparent=True,bbox_inches='tight')   
    training_logs.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-batch', type=int, action="store", dest='batch', default=10)
    parser.add_argument('-type',type=str, action="store", dest='runningType', default='noAug')
    parser.add_argument('-model_name',type=str, action="store", dest='model_name', default='resnet50')
    parser.add_argument('-epochs',type=int, action="store", dest='epochs', default=100)
    parser.add_argument('-lr',type=float, action="store", dest='lr', default=0.001)
    args = parser.parse_args()
    main(args)
