#for cv2 loading takes most of the time
#hence preloading required, to accomodate memory for preloading resolution of image files
#reduced to 2K from 4K

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
from torch.utils.tensorboard import SummaryWriter


#image size passing to the CNN
#traindir = "C:\\yolo\\darknet\\build\\darknet\\x64\\data\\123\\"
#traindir = "C:\\yolo\\darknet\\build\\darknet\\x64\\data\\new_training\\"
traindir = "H:\\new_training_reduced\\"
logdir = "C:\\yolo\pytorch-yolo-v3\\"
epochs = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4 
momentum = 0.9
rect_size = 224
image_size = [rect_size, rect_size]
batch_size = 8#4 #batch_size for training
log_step = 450 #990 #450
comp_techs = 3
path_saved_model = "C:\\temp\\sibu_check_model"
save_model_step = 1 #10


org_reso = 3840#1920
#how to decide these values (need to figure it out)
det_reso = [3840, 2160]
#det_reso = [1920, 1080]
best_jpeg = 120#8
best_qp = 4
bw = 5
comm_time = 60
tx_size = (bw * 1024 * 1024 * comm_time)/8000
#folder list
trainset = [2,3,9,11,17,21,24,25,27,29,30,32,35,38]
train_fol_size = [251,264,441,300,144,360,360,263,420,288,121,170,384,210]
#for testing 
#trainset = [2, 3, 9, 11]
#train_fol_size = [1, 1, 1, 3]


#initialization for detection
FILE_DIR = ""
predict_dominant_class = False #True
dominant_cls_list = []     

#dont need to call this multiple times
#images = args.images #don't need to store the detection
test_batch_size = 1 #int(args.bs)
confidence = 0.5 #float(args.confidence)
nms_thesh = 0.4 #float(args.nms_thresh)
CUDA = torch.cuda.is_available()
num_classes = 80
classes = load_classes('data/coco.names')
cfgfile = "cfg//yolov3.cfg"
weightsfile = "yolov3.weights"

#Set up the neural network
#print("Loading network.....")
model = Darknet(cfgfile)
model.load_weights(weightsfile)
#print("Network successfully loaded")

if CUDA:
	model.cuda()
#Set the model in evaluation mode
model.eval()

#print("\n start of loading \n")
#load all training images
total_train_imgs = sum(train_fol_size)
train_image_list = []
"""
for i in range(total_train_imgs):
    srcfile  = traindir + str(i) +"\\pic_" + str(i) + "_org.png"
    img = cv2.imread(srcfile, cv2.IMREAD_UNCHANGED)
    train_image_list.append(img)
    print(i)
"""
print("\n ...... loading finished ........ \n")
writer = SummaryWriter('log_training_dir')


def detect(det_file, reso, logfile):
    model.net_info["height"] = reso
    inp_dim = reso
    images = det_file #image file on which detection will be performed
    f = open(FILE_DIR + logfile + ".txt","w+")
    #print("\n inside detect function \n")

    #we dont need to assert the sizes as its always > 32 and divisible by 32
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        #print ("No file or directory with the name {}".format(images))
        exit()
    #print(imlist)
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    leftover = 0
    
    if (len(im_dim_list) % test_batch_size):
        leftover = 1
        
        
    if test_batch_size != 1:
        num_batches = len(imlist) // test_batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*test_batch_size : min((i +  1)*test_batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        
    i = 0
    write = False    
    objs = {}      
    for batch in im_batches:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA) 
        
        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        if type(prediction) == int:
            i += 1
            continue          

        prediction[:,0] += i*test_batch_size  
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))

        i += 1
        if CUDA:
            torch.cuda.synchronize()
                      
    
    try:
        output
    except NameError:
        #print("No detections were made")
        return -1
    
    #this part in order to predict one class only    
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
    
    
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    #to #print the output from same function
    out_x = output.cpu().numpy()
    index_pic = (int((str(images).split("\\")[-1]).split("_")[1]))
    for i in range(len(out_x)):
        x = out_x[i]
        f.write("%f %f %f %f %d %d\n" %(x[1],x[2], x[3],x[4], x[-1], index_pic))
    return 0
        
    
    
    #f.close()    
    #torch.cuda.empty_cache()






 

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
        self.curr_index = 0
        self.pic_number = 1
        #self.detmethod = detect
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
        #print("\n\n\n#printing out the model:")
        #print(model)
        #print("\n\n\na summary of input/output for the model:")
        summary(model, (3,image_size[0],image_size[1]),-1, device='cpu')
	
	#run training
    def training_code(self, model):        
        model = copy.deepcopy(model)
        model = model.to(self.device)
        #instead of the criterion I have to define my own loss function
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)#, momentum=self.momentum)
        for epoch in range(self.epochs):  
            #print("\n epoch :", epoch)
            running_loss = 0.0
            start_epoch = time.time()
            log_counter = 1
            for i, data in enumerate(self.train_data_loader):
                inputs, labels = data
                #print("\n the current label is :", labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                ##  Since PyTorch likes to construct dynamic computational graphs, we need to
                ##  zero out the previously calculated gradients for the learnable parameters:
                optimizer.zero_grad()
                # Make the predictions with the model:
                outputs = model(inputs)
                #to get only postive outputs need to apply softmax
                #outputs = torch.nn.functional.softmax(outputs)
                loss = criterion(outputs, labels)
                #loss_scalar = self.myloss(outputs, labels)
                #loss = self.final_loss(outputs, labels, loss_scalar)
                loss.backward()
                optimizer.step()
                ##  Present to the average value of the loss over the past 2000 batches:            
                running_loss += loss.item()
                if i % log_step == log_step-1:    
#                    #print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(2000)))
                    avg_loss = running_loss / float(log_step)
                    print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, avg_loss))
                    running_loss = 0.0
            #after each epoch reinitializing
            self.curr_index = 0
            self.pic_number = 1
            if(epoch%save_model_step==0):
                #print("\n saving the current model \n")
                self.path_saved_model = path_saved_model + "_" + str(epoch)
                self.save_model(model)
        print("\n SP: this epoch takes = %f ms" %(time.time() - start_epoch)*1000)
        #start_epoch = time.time()
        print("\nFinished Training\n")
        self.save_model(model)
    
    def final_loss(self, outputs, labels, loss_scalar):
        temp_loss = torch.mean((outputs.float())**2)
        ##print(temp_loss)
        ##print("\n SE loss: ", temp_loss.item())
        scale = loss_scalar/temp_loss.item()
        ##print(scale)
        loss = temp_loss*scale
        #print("\n final loss value:", loss)
        return loss
    
    def myloss(self, outputs, labels):
        batch_loss = 0.0
        file_index_list = labels.cpu().numpy()
        #writing to tackle higher batch sizes
        for i in range(len(file_index_list)):
            file_index = file_index_list[i]
            #print(outputs, outputs.detach().cpu().numpy()[i])
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
            #print("\n reso_weight = %f, jpeg weight = %f, encode weight = %f" %(reso_weight, jpeg_weight, encode_weight))
            ##print("\n current file index = %d, output[0]= %d" %(file_index, reso_weight))
            infile = traindir + str(file_index) +"\\pic_" + str(file_index) + "_org.png"
            """
            reso = int(detection_reso * reso_weight)
            jpeg_qp = int(best_jpeg/jpeg_weight)
            encoding_qp = int(best_qp/encode_weight)
            """
            ##print("\n myloss(): picname = %s, current resolution is = %d" % (infile, reso))
            #for all three compression
            temp_loss = self.compress(infile, file_index, reso_weight, jpeg_weight, encode_weight)
            #only downsizing
            #temp_loss = self.downsize_compress(infile, reso, jpeg_qp, encoding_qp)
            #print("\n SP: cumulative loss = ", temp_loss)
            #loss = torch.tensor(temp_loss, requires_grad=True) #in order to backpropagate the loss
            batch_loss += temp_loss
        avg_loss = batch_loss/len(file_index_list)
        #print("\n SP: final loss needs to be backpropagate : %f" %(batch_loss/len(file_index_list)))
        return avg_loss
 
       
    
    def compress(self, infile, file_index, reso_weight, jpeg_weight, encode_weight):
        #downsizing
        tempfile = "temp_2.mp4"
        outfile = "temp_2_org.png"
        temp_jpeg = "temp_2.jpg"
        width = int(det_reso[0] * reso_weight)#higher the better
        height = int(det_reso[1] * reso_weight)
        if(width%2>0):
            width+=1
        if(height%2>0):
            height+=1
        """
        encoding_qp = int(best_qp/encode_weight)
        jpeg_qp = int(best_jpeg/jpeg_weight)#higher the better
        """
        dim = (width, height)
        #print(dim)
        img = train_image_list[file_index]
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #JPEG compression
        jpeg_qp = int(best_jpeg*jpeg_weight)#higher the better
        cv2.imwrite(temp_jpeg, resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_qp])
        #encoding
        encoding_qp = int(best_qp/encode_weight)
        cmd_str = "ffmpeg -i "+ temp_jpeg + " -c:v libx264 -qp " + str(encoding_qp) +" -loglevel quiet -y " + tempfile #-loglevel quiet
        os.system(cmd_str)
        """
        cmd_str = "ffmpeg -i " + infile +" -vf scale=" + str(width)+":" + str(height) + " -q:v "+ str(jpeg_qp) +" -c:v libx264 -qp " + str(encoding_qp) + " temp_1.mp4"
        os.system(cmd_str)
        """
        print("\n file_index = %d, current resolution is = %d, jpeg qp = %d, encode qp = %d" % (file_index, width, jpeg_qp, encoding_qp))
        #revert back(need this?)
        vidObj = cv2.VideoCapture(tempfile)
        success, image = vidObj.read()  
        #dim = (3840,2160)
        resized = cv2.resize(image, (det_reso[0],det_reso[1]), interpolation = cv2.INTER_AREA)
        cv2.imwrite(outfile, resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        """
        cmd_str = "ffmpeg -i temp_1.mp4 -vf scale=3840:2160 temp_2_org.png"
        os.system(cmd_str)
        """
        acc_loss = self.find_acc_loss(infile, outfile)
        #checking the size if its more then +1 if less than or equal to then 0 loss
        #later modification needed on how two losses need to propagate
        acc_loss += self.check_size(tempfile)
        #deleting
        """
        os.system("rm " + temp_jpeg)
        os.system("rm " + tempfile)
        #os.system("rm temp.png")
        #os.system("rm temp_1.png")
        os.system("rm " + outfile)
        """
        return acc_loss
        
    
    #in order to speedup the execution
    def find_acc_loss(self, orgfile, testfile):
        #ind = detect(orgfile, 1024, "log_org")
        #alternative way
        pic_index = int((orgfile.split("\\")[-1]).split("_")[1])+1
        if(self.pic_number > train_fol_size[self.curr_index]):
            self.pic_number = 1
            if(self.curr_index < len(train_fol_size)):
                self.curr_index+=1
        log_org_file = "bbox_drone_" + str(trainset[self.curr_index]) + "\\bbox_org\\log_bbox_org_" + str(self.pic_number) + ".txt"
        self.pic_number+=1
        ind = detect(testfile, 1024, "log_test")
        #acc_loss, mismatch_org, total_org, total, correct_detection = cal_mul.cal_iou("log_org.txt", "log_test.txt")
        #print("SP: current pic_index :%d log_org_file:%s "%(pic_index,log_org_file))
        acc_loss, mismatch_org, total_org, total, correct_detection = cal_mul.cal_iou(log_org_file, "log_test.txt")
        print("\n SP: the accuracy loss only = %f"%(1.0 - acc_loss))
        return 1.0 - acc_loss
    
    def check_size(self,outfile):
        statinfo = os.stat(outfile)
        if(statinfo.st_size > tx_size):
            #print("\n Warning: Current filesize more than estimated \n")
            return 1.0 #add large penalty
        else:
            #print("\n File size is OK \n")
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
    #print("\n .... after traindataloader .......\n")
    return train_loader
	
def main():
    train_loader = load_custom_dataset(traindir)
    #print("\n after trainloader ", len(train_loader))
    detnet = Detection_net(train_loader, epochs, device, momentum, batch_size, learning_rate, path_saved_model)
    #print("\n after creating detnet \n")
    net = detnet.Net()
    #print("\n after creating net for detnet \n")
    detnet.show_network_summary(net)
    #print("\n after network summary \n")
    p=psutil.Process()
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    detnet.training_code(net)


if __name__== "__main__":
  main()
	




