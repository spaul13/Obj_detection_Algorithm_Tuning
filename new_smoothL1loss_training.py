from __future__ import division
import random
import numpy as np
import torch
import os, sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary           
import numpy as np
import re
import math
import copy
import matplotlib.pyplot as plt
import calculate_iou_2obj as cal_mul
from PIL import Image

#for detection
import time
from torch.autograd import Variable
import cv2 
from util import *
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import pickle as pkl
import itertools
import psutil

#image size passing to the CNN

traindir = "reduced_regression_training_fol\\"
logdir = "C:\\yolo\pytorch-yolo-v3\\"
epochs = 5000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4 
momentum = 0.9
rect_size = 224
image_size = [rect_size, rect_size]
batch_size = 16
log_step = 75 
path_saved_model = "C:\\temp\\sibu_0406_smoothL1loss_model"
save_model_step = 500 #10
org_reso = 3840
total_configs = 107

class Detection_net(nn.Module):
    def __init__(self, trainloader, epochs, device, momentum, batch_size, learning_rate, path_saved_model):
        super(Detection_net, self).__init__()
        self.train_data_loader = trainloader
        self.epochs = epochs
        self.device = device
        self.momentum = momentum
        self.batch_size = batch_size
        self.learning_rate = learning_rate	
        self.path_saved_model = path_saved_model
        #self.detmethod = detect
    class Net(nn.Module):
        def __init__(self):
            super(Detection_net.Net, self).__init__()
            self.conv_seqn = nn.Sequential(
                # Conv Layer block 1:
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fc_seqn = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(100352, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1)
            )
        def forward(self, x):
            x = self.conv_seqn(x)
            # flatten
            x = x.view(x.size(0), -1)
            x = self.fc_seqn(x)
            return x

    #save the model
    def save_model(self, model):
        torch.save(model.state_dict(), self.path_saved_model)

    #get the summary of the model
    def show_network_summary(self, model):
        summary(model, (3,image_size[0],image_size[1]),-1, device='cpu')
	
	#run training
    def training_code(self, model):        
        model = copy.deepcopy(model)
        model = model.to(self.device).float()
        #instead of the criterion I have to define my own loss function
        #criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss().cuda()
        #criterion = nn.CrossEntropyLoss()#Combination of nn.LogSoftmax() and nn.NLLLoss()
        #optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):  
            print("\n epoch :", epoch)
            running_loss = 0.0
            start_epoch = time.time()
            for i, data in enumerate(self.train_data_loader):
                inputs, labels = data
                #print("\n the current label is :", labels)
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device).float()
                ##  Since PyTorch likes to construct dynamic computational graphs, we need to
                ##  zero out the previously calculated gradients for the learnable parameters:
                optimizer.zero_grad()
                # Make the predictions with the model:
                outputs = model(inputs)
                #to get only postive outputs need to apply softmax
                #outputs = torch.nn.functional.softmax(outputs)
                #outputs = torch.clamp(outputs, min=0.1, max=1)
                labels = torch.unsqueeze(labels, 1)
                #print(labels,outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                ##  Present to the average value of the loss over the past 2000 batches:            
                running_loss += loss.item()
                if i % log_step == log_step-1:    
#                    #print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(2000)))
                    avg_loss = running_loss / float(i+1)
                    print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, avg_loss))
                    running_loss = 0.0
            #after each epoch reinitializing
            self.curr_index = 0
            self.pic_number = 1
            if(epoch%save_model_step==0):
                #print("\n saving the current model \n")
                self.path_saved_model = path_saved_model + "_" + str(epoch)
                self.save_model(model)
            #print("\n SP: this epoch takes = %f ms" %(time.time() - start_epoch)*1000)
        print("\nFinished Training\n")
        self.save_model(model)

def load_custom_dataset(traindir):
    transform = tvt.Compose([tvt.ToTensor(),tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #train_dataset = torchvision.datasets.ImageFolder(traindir,transform=transform)
    train_dataset = torchvision.datasets.ImageFolder(traindir,tvt.Compose([tvt.RandomResizedCrop(image_size[0]),transform]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = 8)
    #print("\n .... after traindataloader .......\n")
    return train_loader

def main():
    train_loader = load_custom_dataset(traindir)
    print(device)
    print("\n after trainloader ", len(train_loader))
    detnet = Detection_net(train_loader, epochs, device, momentum, batch_size, learning_rate, path_saved_model)
    print("\n after creating detnet \n")
    net = detnet.Net()
    print("\n after creating net for detnet \n")
    detnet.show_network_summary(net)
    print("\n after network summary \n")
    detnet.training_code(net)


if __name__== "__main__":
  main()


