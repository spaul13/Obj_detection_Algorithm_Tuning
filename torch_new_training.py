import random
import numpy as np
import torch
import os, sys,os.path
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

#image size passing to the CNN
#traindir = "C:\\yolo\\darknet\\build\\darknet\\x64\\data\\123\\"
traindir = "C:\\yolo\\darknet\\build\\darknet\\x64\\data\\new_training\\"
logdir = "C:\\yolo\pytorch-yolo-v3\\"
epochs = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4 
momentum = 0.9
rect_size = 224
image_size = [rect_size, rect_size]
batch_size = 8
log_step = 1200
comp_techs = 3
path_saved_model = "C:\\temp\\sibu_new_model"
save_model_step =10


org_reso = 3840
#how to decide these values (need to figure it out)
detection_reso = 3840 #1024
best_jpeg = 8
best_qp = 6
bw = 5
comm_time = 60
tx_size = (bw * 1024 * 1024 * comm_time)/8000
 

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
    class Net(nn.Module):
        def __init__(self):
            super(Detection_net.Net, self).__init__()
            self.conv_seqn = nn.Sequential(
                # Conv Layer block 1:
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=8, stride=8),
            )
            self.fc_seqn = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(50176, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(256, comp_techs)
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
        print("\n\n\nprinting out the model:")
        print(model)
        print("\n\n\na summary of input/output for the model:")
        summary(model, (3,image_size[0],image_size[1]),-1, device='cpu')
	
	#run training
    def training_code(self, model):        
        model = copy.deepcopy(model)
        model = model.to(self.device)
        #instead of the criterion I have to define my own loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        for epoch in range(self.epochs):  
            print("\n epoch :", epoch)
            running_loss = 0.0
            for i, data in enumerate(self.train_data_loader):
                inputs, labels = data
                print("\n the current label is :", labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                ##  Since PyTorch likes to construct dynamic computational graphs, we need to
                ##  zero out the previously calculated gradients for the learnable parameters:
                optimizer.zero_grad()
                # Make the predictions with the model:
                outputs = model(inputs)
                #to get only postive outputs need to apply softmax
                #outputs = torch.nn.functional.softmax(outputs)
                #loss = criterion(outputs, labels)
                loss = self.myloss(outputs, labels)
                loss.backward()
                optimizer.step()
                ##  Present to the average value of the loss over the past 2000 batches:            
                running_loss += loss.item()
                if i % log_step == log_step-1:    
#                    print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(2000)))
                    avg_loss = running_loss / float(log_step)
                    print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, avg_loss))
                    running_loss = 0.0
            if(epoch%save_model_step==0):
                print("\n saving the current model \n")
                self.path_saved_model = path_saved_model + "_" + str(epoch)
                self.save_model(model)        
        print("\nFinished Training\n")
        self.save_model(model)
    
    def myloss(self, outputs, labels):
        batch_loss = 0.0
        file_index_list = labels.cpu().numpy()
        #writing to tackle higher batch sizes
        for i in range(len(file_index_list)):
            file_index = file_index_list[i]+1
            print(outputs, outputs.detach().cpu().numpy()[i])
            current_output = outputs.detach().cpu().numpy()[i]
            #"""
            #Need to detach first as its part of computational graph, requires_grad = True
            #Problem: output of the network can be negative, how to restrict it to be positive
            reso_weight = abs(current_output[0])#to get the positive value
            if(reso_weight<0.1):
                reso_weight*=10
            jpeg_weight = abs(current_output[1])
            if(jpeg_weight < 0.1):
                jpeg_weight*=10
            encode_weight = abs(current_output[2])
            if(encode_weight < 0.1):
                encode_weight*=10
            #"""
            #with softmax the problem is three outputs are with [0.3-0.35] making the range narrow
            #need to find other variants
            #reso_weight, jpeg_weight, encode_weight = current_output[0], current_output[1], current_output[2]
            print("\n reso_weight = %f, jpeg weight = %f, encode weight = %f" %(reso_weight, jpeg_weight, encode_weight))
            print("\n current file index = %d, output[0]= %d" %(file_index, reso_weight))
            infile = traindir + str(file_index) +"\\pic_" + str(file_index) + "_org.png"
            reso = int(detection_reso * reso_weight)
            jpeg_qp = int(best_jpeg/jpeg_weight)
            encoding_qp = int(best_qp/encode_weight)
            print("\n myloss(): picname = %s, current resolution is = %d" % (infile, reso))
            #for all three compression
            temp_loss = self.compress(infile, reso, jpeg_qp, encoding_qp)
            #only downsizing
            #temp_loss = self.downsize_compress(infile, reso, jpeg_qp, encoding_qp)
            print("\n SP: cumulative loss = ", temp_loss)
            #loss = torch.tensor(temp_loss, requires_grad=True) #in order to backpropagate the loss
            batch_loss += temp_loss
        avg_loss = torch.tensor(batch_loss/len(file_index_list), requires_grad=True)
        print("\n SP: final loss needs to be backpropagate : %f" %(batch_loss/len(file_index_list)))
        return avg_loss
        
    #only downsizing
    def downsize_compress(self, infile, reso, jpeg_qp, encoding_qp):
        print("\n picname = %s, current resolution is = %d" % (infile, reso))
        #downsizing at client
        cmd_str = "ffmpeg -i " + infile +" -vf scale=" + str(int(reso)) +":-1 " + "temp.png"
        os.system(cmd_str)
        #upscaling at server
        #cmd_str = "ffmpeg -i temp.png -vf scale=" + str(int(org_reso)) +":-1 " + "temp_1_org.png"
        cmd_str = "ffmpeg -i temp.png -vf scale=1024:576 temp_1_org.png"
        os.system(cmd_str)
        acc_loss = self.find_acc_loss(infile, "temp_1_org.png")
        #checking the size if its more then +1 if less than or equal to then 0 loss
        #later modification needed on how two losses need to propagate
        acc_loss += self.check_size("temp.png")
        #deleting
        os.system("rm temp.png")
        os.system("rm temp_1_org.png")
        return acc_loss
       
    
    def compress(self, infile, reso, jpeg_qp, encoding_qp):
        print("\n picname = %s, current resolution is = %d, jpeg qp = %d, encode qp = %d" % (infile, reso, jpeg_qp, encoding_qp))
        #downsizing
        cmd_str = "ffmpeg -i " + infile +" -vf scale=" + str(int(reso)) +":-1 " + "temp.png"
        os.system(cmd_str)
        #JPEG compression
        cmd_str = "ffmpeg -i temp.png -q:v " + str(jpeg_qp) + " temp.jpg"
        os.system(cmd_str)
        #encoding
        cmd_str = "ffmpeg -i temp.jpg -c:v libx264 -qp " + str(encoding_qp) +" temp.mp4"
        os.system(cmd_str)
        #revert back(need this?)
        cmd_str = "ffmpeg -i temp.mp4 temp_1.png"
        os.system(cmd_str)
        #cmd_str = "ffmpeg -i temp_1.png -vf scale=1024:576 temp_1_org.png"
        cmd_str = "ffmpeg -i temp_1.png -vf scale=3840:2160 temp_1_org.png"
        os.system(cmd_str)
        acc_loss = self.find_acc_loss(infile, "temp_1_org.png")
        #checking the size if its more then +1 if less than or equal to then 0 loss
        #later modification needed on how two losses need to propagate
        acc_loss += self.check_size("temp.mp4")
        #deleting
        os.system("rm temp.jpg")
        os.system("rm temp.mp4")
        os.system("rm temp.png")
        os.system("rm temp_1.png")
        os.system("rm temp_1_org.png")
        return acc_loss
    
    def find_acc_loss(self, orgfile, testfile):
        cmd_str = "python detect.py --images " + orgfile + " --reso 1024 --file_name log_org"
        os.system(cmd_str)
        print("\n detection on original file is done \n")
        cmd_str = "python detect.py --images " + testfile + " --reso 1024 --file_name log_test"
        os.system(cmd_str)
        print("\n detection on compressed file is done \n")
        acc_loss, mismatch_org, total_org, total, correct_detection = cal_mul.cal_iou("log_org.txt", "log_test.txt")
        print("\n SP: the accuracy loss only = %f"%(1.0 - acc_loss))
        return 1.0 - acc_loss
    
    def check_size(self,outfile):
        statinfo = os.stat(outfile)
        if(statinfo.st_size > tx_size):
            print("\n Warning: Current filesize more than estimated \n")
            return 1.0 #add large penalty
        else:
            print("\n File size is OK \n")
            return 0.0
        
        
    

	
	#run testing
	#def run_testing(self, model):
	

#loading the images
#keep all images inside a folder
def load_custom_dataset(traindir):
    transform = tvt.Compose([tvt.ToTensor(),tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #train_dataset = torchvision.datasets.ImageFolder(traindir,transform=transform)
    train_dataset = torchvision.datasets.ImageFolder(traindir,tvt.Compose([tvt.RandomResizedCrop(image_size[0]),transform]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    print("\n .... after traindataloader .......\n")
    return train_loader
	
def main():
    train_loader = load_custom_dataset(traindir)
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
	




