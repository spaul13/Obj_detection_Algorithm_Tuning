from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools
counter = 0
#f = open("bbox_image_jpg.txt","w+") #352, 480, 608, 704
FILE_DIR = ""
predict_dominant_class = False #True
picnumber = []
piclist = []
dominant_cls_list = []
        

#dont need to call this multiple times
#images = args.images #don't need to store the detection
batch_size = 1 #int(args.bs)
confidence = 0.5 #float(args.confidence)
nms_thesh = 0.4 #float(args.nms_thresh)


CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes('data/coco.names')
cfgfile = "cfg//yolov3.cfg"
weightsfile = "yolov3.weights"

#Set up the neural network
print("Loading network.....")
model = Darknet(cfgfile)
model.load_weights(weightsfile)
print("Network successfully loaded")

if CUDA:
	model.cuda()
#Set the model in evaluation mode
model.eval()
reso = 1024
logfile = "log_try"
det_file = "C:\\yolo\\darknet\\build\\darknet\\x64\\data\\123\\1\\pic_1_org.png"
	



def detect(det_file,reso,logfile):
#if __name__ ==  '__main__':
    model.net_info["height"] = reso
    inp_dim = reso
    images = det_file #image file on which detection will be performed
    f = open(FILE_DIR + logfile + ".txt","w+")

    #we dont need to assert the sizes as its always > 32 and divisible by 32
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
    print(imlist)
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    leftover = 0
    
    if (len(im_dim_list) % batch_size):
        leftover = 1
        
        
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
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

        prediction[:,0] += i*batch_size
        
    
            
          
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
        print("No detections were made")
        exit()
    
    #this part in order to predict one class only    
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
    
    
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    #to print the output from same function
    out_x = output.cpu().numpy()
    index_pic = (int((str(images).split("\\")[-1]).split("_")[1]))
    for i in range(len(out_x)):
        x = out_x[i]
        f.write("%f %f %f %f %d %d\n" %(x[1],x[2], x[3],x[4], x[-1], index_pic))
    #print(x.shape(), x[0])
    
    
def write(x, batches, results, images):
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    index_pic = (int((str(images).split("\\")[-1]).split("_")[1]))
    f.write("%f %f %f %f %d %d\n" %(x[1].item(),x[2].item(), x[3].item(),x[4].item(), cls, index_pic))
    if((predict_dominant_class) and (cls == dominant_cls_list[0])):
        #print ('x1,y1: ', x[1].item(),x[2].item())
        #print ('x2,y2: ', x[3].item(),x[4].item())
        #print(picnumber)
        f.write("%f %f %f %f %d %d\n" %(x[1].item(),x[2].item(), x[3].item(),x[4].item(), cls, index_pic))#uncomment
    else:
        index_pic = (int((str(images).split("\\")[-1]).split("_")[1]))
        f.write("%f %f %f %f %d %d\n" %(x[1].item(),x[2].item(), x[3].item(),x[4].item(), cls, index_pic))
    img = results[int(x[0])]
    #return img
    
"""           
list(map(lambda x: write(x, im_batches, orig_ims), output))

f.close()    
torch.cuda.empty_cache()
"""
    

def main():
    detect(det_file,reso,logfile)
    #list(map(lambda x: write(x, im_batches, orig_ims, images), output))
    #write(x, im_batches, orig_ims)

if __name__ == "__main__":
	main()

	
	