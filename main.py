
from . import A,B
from A.Train_A import Train_A,plot_loss
from B.Train_B import Train_B
from A.Test_A import test_A
from B.Test_B import test_B
from A.Model_A import Pneumonia_resnet
from B.Model_B import ResnetPath3

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import resnet50

import medmnist
from medmnist import INFO,PneumoniaMNIST,PathMNIST
from medmnist.evaluator import Evaluator


data_flag_A,data_flag_B = 'pneumoniamnist','pathmnist'

# Pre-trained Feature Extractor
pretrained = resnet50(pretrained=True)

def main(data_flag,pretrained,BATCH_SIZE,lr,device):
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    # ================== Task A: Binary Classification ==================
    if data_flag == 'pneumoniamnist':
        # Data Augmentation and Preprocessing
        data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-20,20)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
            ])
        train_dataset = PneumoniaMNIST(split='train', transform=data_transform, root='./Datasets/')
        val_dataset= PneumoniaMNIST(split='val', transform=data_transform, root='./Datasets/')
        test_dataset = PneumoniaMNIST(split='test', transform=data_transform, root='./Datasets/')
        # Load Data
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loader_at_eval = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # Load Model
        model = Pneumonia_resnet(pretrained,num_classes=n_classes)
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Start Training
        NUM_EPOCHS = 15
        print("================== Start Training A ==================")
        Train_A(model,train_loader,train_loader_at_eval,test_loader,NUM_EPOCHS,criterion,optimizer,device)

    # ================== Task B: Multi-class Classification ==================
    else:
        # Data Augmentation and Preprocessing
        data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-20,20)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
            ])
        train_dataset = PathMNIST(split='train', transform=data_transform, root='./Datasets/')
        val_dataset= PathMNIST(split='val', transform=data_transform, root='./Datasets/')
        test_dataset = PathMNIST(split='test', transform=data_transform, root='./Datasets/')
        # Load Data
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loader_at_eval = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # Load Model
        model = ResnetPath3(pretrained,num_classes=n_classes)
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Start Training
        NUM_EPOCHS = 30
        print("================== Start Training B ==================")
        Train_B(model,train_loader,train_loader_at_eval,test_loader,NUM_EPOCHS,criterion,optimizer,device)


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64
    lr = 0.0001

    # Task A: Binary Classification
    main(data_flag_A,pretrained,BATCH_SIZE,lr,device)
    # Task B: Multi-class Classification
    main(data_flag_B,pretrained,BATCH_SIZE,lr,device)