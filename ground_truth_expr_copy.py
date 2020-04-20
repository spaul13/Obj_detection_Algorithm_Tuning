from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
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


import numpy as np
import matplotlib.pyplot as plt
import statistics, os, sys
import calculate_iou_2obj as cal_mul
from matplotlib.cm import get_cmap
import os.path
#reso_list = [3200, 2560, 2048, 1440, 960, 768, 640, 480, 320, 160]
reso_list = [2048, 1440, 960, 768, 640, 480, 320, 160]
#reso_list = [160]
#qp_list = np.linspace(0,40,11)
qp_list = np.linspace(12,40,8)
#qp_list = [28,32,36, 40]
#qp_list = np.linspace(16,40,7)
MAX_NUM = 263#360#144 #264 #180#145#384#325#300#170#184#121#288#199#420#261#263#360#390#330#300#144#420#300#230#441#300#230#441 #420 #350 #144 216 #264 #250
MIN_NUM = 1
dir_name = "bbox_drone_25\\"
jpeg_enabled = True
org_res = 3840

#sz_list = []
#mech_list = []

fs = 30
count_acc = []


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


def detect(det_file, reso, logfile):
    model.net_info["height"] = reso
    inp_dim = reso
    images = det_file #image file on which detection will be performed
    f = open(FILE_DIR + logfile + ".txt","w+")
    print("\n inside detect function \n")

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
        #print("detections = %f %f %f %f %d %d\n" %(x[1],x[2], x[3],x[4], x[-1], index_pic))
        f.write("%f %f %f %f %d %d\n" %(x[1],x[2], x[3],x[4], x[-1], index_pic))
    return 0
        
    
    
    #f.close()    
    #torch.cuda.empty_cache()





def Downsize(infol, outfol, reso):
    for i in range(MIN_NUM, MAX_NUM+1):
        infile = infol + "\pic_" + str(i) + "_org.png"
        outfile = outfol + "\pic_" + str(i) + "_temp.png"
        #Downscaling (at client side)
        cmd_str = "ffmpeg -i " + infile +" -vf scale=" + str(int(reso)) +":-1 " + outfile
        os.system(cmd_str)
        #Upscaling (Server side)
        outfile_1 = outfol + "\pic_" + str(i) + "_org.png"
        cmd_str = "ffmpeg -i " + outfile +" -vf scale=" + str(org_res) +":-1 " + outfile_1
        os.system(cmd_str)
        #deletion
        #cmd_str = "rm " + outfile
        #os.system(cmd_str)
		

def convert_jpeg(infol, outfol, order):
    for i in range(MIN_NUM, MAX_NUM+1):
        if(order==1):
            infile = infol + "\pic_" + str(i) + "_temp.png"
        else:
            infile = infol + "\pic_" + str(i) + "_org.png"
        outfile = outfol + "\pic_" + str(i) + "_temp.jpg"
        #jpeg compression on client side
        cmd_str = "ffmpeg -i " + infile + "  " + outfile
        os.system(cmd_str)
        #Upscaling (Server side)
        outfile_1 = outfol + "\pic_" + str(i) +"_org.jpg" 
        cmd_str = "ffmpeg -i " + outfile +" -vf scale=" + str(org_res) +":-1 " + outfile_1
        os.system(cmd_str)

def encode(infol, outfol, qp, jp):
    for i in range(MIN_NUM, MAX_NUM+1):
        outfile = outfol + "\pic_" + str(i) + "_org.mp4"
        if(jp==0):
            infile = infol + "\pic_" + str(i) + "_org.png"
        else:
            infile = infol + "\pic_" + str(i) + "_org.jpg"
        cmd_str = "ffmpeg -i " + infile +" -c:v libx264 -qp " + str(qp) +"  " + outfile
        os.system(cmd_str)
        
def size(infol, type, frame_id): #type = 0 
    sum = 0
    #for i in range(MIN_NUM, MAX_NUM+1):
    for i in range(frame_id, frame_id+1):
        if(type==0):
            outfile = infol + "\pic_" + str(i) + "_temp.png"
        elif(type==1):
            outfile = infol + "\pic_" + str(i) + "_temp.jpg"
        elif(type==2):
            outfile = infol + "\pic_" + str(i) + "_temp.mp4"
        else: #this is only encoding or jpeg+encoding
            outfile = infol + "\pic_" + str(i) + "_org.mp4"
        statinfo = os.stat(outfile)
        sum+=statinfo.st_size
    
    #size_kb = sum/(MAX_NUM*1024)
    size_kb = sum/1024
    #print(size_kb)
    return size_kb
        
def down_encode(infol, outfol, qp, jp):
    for i in range(MIN_NUM, MAX_NUM+1):
        outfile = outfol + "\pic_" + str(i) + "_temp.mp4"
        if(jp==0):
            infile = infol + "\pic_" + str(i) + "_temp.png"
        else:
            infile = infol + "\pic_" + str(i) + "_temp.jpg"
        #encoding
        cmd_str = "ffmpeg -i " + infile +" -c:v libx264 -qp " + str(qp) +"  " + outfile
        os.system(cmd_str)        
        #decoding
        cmd_str = "ffmpeg -i "  + outfile + " temp.png"
        if(jp==0):
            temp_file = infol + "\\temp.png"
            cmd_str = "ffmpeg -i "  + outfile + "  " + temp_file
            os.system(cmd_str)
            outfile_1 = outfol + "\pic_" + str(i) + "_org.png"
            cmd_str = "ffmpeg -i " + temp_file  + " -vf scale=" + str(org_res) +":-1 " + outfile_1
            os.system(cmd_str)
            os.system("rm " + temp_file)
        else:
            temp_file = infol + "\\temp.jpg"
            cmd_str = "ffmpeg -i "  + outfile + "  " + temp_file
            os.system(cmd_str)
            outfile_1 = outfol + "\pic_" + str(i) + "_org.jpg"
            cmd_str = "ffmpeg -i " + temp_file  + " -vf scale=" + str(org_res) +":-1 " + outfile_1
            os.system(cmd_str)
            os.system("rm " + temp_file)


def perf_detection(fol_in):
    if("encode" in fol_in):
        temp = fol_in.split("\\")[-1]
        print(temp)
       

def count(list1, mech_list, lb, ub, fol, bw_str, frame_id):
    #return len(list(x for x in list1 if lb <= x <= ub)) 
    c = 0 
    ret_list = []
    bw_old_str = "bw_" + str(ub)
    for x in range(len(list1)): 
        # condition check 
        if list1[x]>= lb and list1[x]< ub: 
            #print(mech_list[x])
            #start of detection
            fol_in = fol+mech_list[x]
            temp_1 = fol_in.split("\\")[-1]
            ret_list.append(bw_str + "\\" + temp_1)
            
            
            
            #print(fol_in)
            temp = fol_in.split("\\")[-2]
            temp_1 = fol_in.split("\\")[-1]
            #print(temp)
            print(temp, temp_1)
            print("\n frame_id = %d, current configuration : %s" %(frame_id,temp_1))
            #in order to write all the configurations into a txt file
            #wfile.write("%s\n"%temp)
            #print(temp)
            
            if(c>=0):
                #first checks whether logfile already exists or not
                outfile_old = dir_name + "bw_"+str(1) + "\\" + temp_1 + "_" + str(frame_id)+".txt"
                outfile_new = dir_name + bw_str + "\\" + temp_1 + "_" + str(frame_id) + ".txt"
                if(os.path.exists(outfile_old)):
                    #print("\n found 1 \n")
                    os.system("scp " + outfile_old + " " + outfile_new)
                elif(os.path.exists(dir_name + "bw_"+str(2) + "\\" + temp_1 + "_" + str(frame_id)+".txt")):
                    #print("\n found 2 \n")
                    outfile_old = dir_name + "bw_"+str(2) + "\\" + temp_1 + "_" + str(frame_id)+".txt"
                    os.system("scp " + outfile_old + " " + outfile_new)
                elif(os.path.exists(dir_name + "bw_"+str(5) + "\\" + temp_1 + "_" + str(frame_id)+".txt")):
                    #print("\n found 5 \n")
                    outfile_old = dir_name + "bw_"+str(5) + "\\" + temp_1 + "_" + str(frame_id)+".txt"
                    os.system("scp " + outfile_old + " " + outfile_new)
                elif(os.path.exists(dir_name + "bw_"+str(10) + "\\" + temp_1 + "_" + str(frame_id)+".txt")):
                    #print("\n found 10 \n")
                    outfile_old = dir_name + "bw_"+str(10) + "\\" + temp_1 + "_" + str(frame_id)+".txt"
                    os.system("scp " + outfile_old + " " + outfile_new)
                else:
                    print("\n SP_0417: Inside Else \n")
                    if((temp == "jpeg_encode") or (temp =="encode")):
                        for i in range(frame_id, frame_id+1):
                            infile = fol_in + "\\pic_" + str(i) + "_org.mp4"
                            outfile = dir_name+bw_str + "\\" + temp_1 + "_" + str(i)
                            cmd_str = "python video_demo.py --video " + infile +" --reso 1024 --file_name " + outfile
                            #ind = detect(infile, 1024, outfile)
                            #cmd_str = "python detect.py --images " + infile + " --file_name " + outfile #for tiny-yolo only
                            print(cmd_str)
                            os.system(cmd_str)
               
                    elif("jpeg" in fol_in):
                        print("inside elif")
                        temp_1 = fol_in.split("\\")[-1]
                        for i in range(frame_id,frame_id+1):
                            infile = fol_in + "\\pic_" + str(i) + "_org.jpg"
                            outfile = dir_name+bw_str + "\\" + temp_1 + "_" + str(i)
                            ind = detect(infile, 1024, outfile)
                            #cmd_str = "python detect.py --images " + infile + " --reso 1024 --file_name " + outfile
                            #cmd_str = "python detect.py --images " + infile + " --file_name " + outfile #for tiny-yolo only
                            #print(cmd_str)
                            #os.system(cmd_str)
                    else:
                        print("inside else")
                        temp_1 = fol_in.split("\\")[-1]
                        for i in range(frame_id,frame_id+1):
                            infile = fol_in + "\\pic_" + str(i) + "_org.png"
                            outfile = dir_name+bw_str + "\\" + temp_1 + "_" + str(i)
                            ind = detect(infile, 1024, outfile)
                            #cmd_str = "python detect.py --images " + infile + " --reso 1024 --file_name " + outfile
                            #cmd_str = "python detect.py --images " + infile + " --file_name " + outfile #for tiny-yolo only
                            #print(cmd_str)
                            #os.system(cmd_str)
            
                    
            #end of detection    
            #"""        
            c+= 1
    
    print("\n SP: for frameid = %d: total configurations are %d"%(frame_id, c))
    print(ret_list)
    #return ret_list
    #return c 


def main():
    print(len(sys.argv))
    total_accuracy = []
    if (len(sys.argv) < 2):
        print("Usage: program image_directory")
    else:
        temp_fol = sys.argv[1]
        print(str(len(sys.argv)) + "," + str(temp_fol))
        list2 = []
        #for raw original frames
        """
        directory = sys.argv[2]
        MAX_NUM = int(sys.argv[3])
        os.system("mkdir " + directory +"\\bbox_training")
        for i in range(1, MAX_NUM+1):
            infile = temp_fol + "\\pic_" + str(i) + "_org.png"
            #outfile = directory + "bbox_org\\log_bbox_org_" + str(i)
            #outfile = directory + "bbox_tiny\\log_bbox_tiny_" + str(i)
            cmd_str = "python detect.py --images " + infile + " --reso 1024 --file_name " + outfile
            #detection on mobile device
            #cmd_str = "python detect.py --images " + infile + " --reso 416 --file_name " + outfile
            print(cmd_str)
            os.system(cmd_str)
        """
        
        #this part is to generated multiple compressed images
        """
        #Downsizing
        for i in range(len(reso_list)):
            out_fol =  str(sys.argv[1]) + "\\downsize\\reso_" + str(reso_list[i])
            os.system("mkdir " + out_fol)
            Downsize(temp_fol, out_fol, reso_list[i])
        
        #JPEG conversion
        out_fol = str(sys.argv[1]) + "\\jpeg"    
        os.system("mkdir " + out_fol)
        convert_jpeg(temp_fol, out_fol, 0)
        
        #Downsize_jpeg
        for i in range(len(reso_list)):
            out_fol =  str(sys.argv[1]) + "\\downsize_jpeg\\reso_" + str(reso_list[i]) + "_jpeg"
            in_fol = str(sys.argv[1]) + "\\downsize\\reso_" + str(reso_list[i])
            os.system("mkdir " + out_fol)
            convert_jpeg(in_fol, out_fol, 1)
        
        #PNG(full_scale+encode)
        for i in range(len(qp_list)):
            out_fol =  str(sys.argv[1]) + "\\encode\\encode_" + str(int(qp_list[i]))
            os.system("mkdir " + out_fol)
            encode(temp_fol, out_fol, qp_list[i], 0)
     
        #JPEG(full_scale+encode)    
        for i in range(len(qp_list)):
            out_fol =  str(sys.argv[1]) + "\\jpeg_encode\\jpeg_encode_" + str(int(qp_list[i]))
            in_fol = str(sys.argv[1]) + "\\jpeg"
            os.system("mkdir " + out_fol)
            encode(in_fol, out_fol, qp_list[i], 1)
        
        
        #Downscale+Encode
        for i in range(len(reso_list)):
            for j in range(len(qp_list)):
                out_fol =  str(sys.argv[1]) + "\\downsize_encode\\" + "reso_" + str(reso_list[i]) + "_encode_" + str(int(qp_list[j]))
                in_fol = str(sys.argv[1]) + "\\downsize\\reso_" + str(reso_list[i])
                os.system("mkdir " + out_fol)
                down_encode(in_fol, out_fol, qp_list[j], 0)
         
        
        #Downscale+JPEG+Encode
        for i in range(len(reso_list)):
            for j in range(len(qp_list)):
                out_fol =  str(sys.argv[1]) + "\\downsize_jpeg_encode\\" + "reso_" + str(reso_list[i]) + "_jpeg_encode_" + str(int(qp_list[j])) 
                in_fol = str(sys.argv[1]) + "\\downsize_jpeg\\reso_" + str(reso_list[i]) + "_jpeg"
                os.system("mkdir " + out_fol)
                down_encode(in_fol, out_fol, qp_list[j], 1)
        
        """
        ub = int(sys.argv[2])
        lb = int(sys.argv[3])
        print(str(lb) + "," +str(ub))
        scale = 8.75 #scale in order to convert the bw to the allowable filesize can be transmitted (in 70 ms), rest is detection time for full yolo
        #scale = 10.25 #for tiny yolo, detection time = 18 ms
        bw_str = "bw_0417_" + str(ub)
        os.system("mkdir " + dir_name + bw_str)
        
        #this part in order to check the generated filesize
        for x in range(MIN_NUM, MAX_NUM+1):
            mech_list, sz_list = [], []
            for i in range(3):
                if(i==0):
                    for j in range(len(reso_list)):
                        temp_file =  str(sys.argv[1]) + "\\downsize\\reso_" + str(reso_list[j])
                        #print("\n ========== \n Downsize (PNG) %d  (in KB) \n ============ \n" %reso_list[j])
                        mech_list.append("\\downsize\\reso_" + str(reso_list[j]))
                        sz_list.append(size(temp_file, 0, x))
                elif(i==1):
                    for j in range(len(reso_list)):
                        temp_file =  str(sys.argv[1]) + "\\downsize_jpeg\\reso_" + str(reso_list[j]) + "_jpeg"
                        #print("\n ========== \n Downsize (JPEG) %d (in KB) \n ============ \n" %reso_list[j])
                        mech_list.append("\\downsize_jpeg\\reso_" + str(reso_list[j]) + "_jpeg")
                        sz_list.append(size(temp_file, 1, x))
                    temp_file = str(sys.argv[1]) + "\\jpeg"
                    #print("\n ========== \n JPEG (full size = 3840) (in KB) \n ============ \n")
                    mech_list.append("\\jpeg")
                    sz_list.append(size(temp_file, 1, x))
                else:
                    for j in range(len(qp_list)):
                        temp_file =  str(sys.argv[1]) + "\\encode\\encode_" + str(int(qp_list[j]))
                        #print("\n ===========\n Encode (QP) %d (in KB) \n ============ \n" %qp_list[j])
                        mech_list.append("\\encode\\encode_" + str(int(qp_list[j])))
                        sz_list.append(size(temp_file, 3, x))
            #PNG_Downsize+Encode
            for i in range(len(reso_list)):
                for j in range(len(qp_list)):
                    #if ((reso_list[i]<960) and (qp_list[j]> 12)):
                    #    continue
                    temp_file =  str(sys.argv[1]) + "\\downsize_encode\\" + "reso_" + str(reso_list[i]) + "_encode_" + str(int(qp_list[j]))
                    #print("\n ===========\n PNG Downsize %d Encode (QP) %d (in KB) \n ============ \n" %(reso_list[i],qp_list[j]))
                    mech_list.append("\\downsize_encode\\reso_" + str(reso_list[i]) + "_encode_" + str(int(qp_list[j])))
                    sz_list.append(size(temp_file, 2, x))
            #PNG_Downsize+JPEG+Encode
            for i in range(len(reso_list)):
                for j in range(len(qp_list)):
                    #if ((reso_list[i]<2048) and (qp_list[j]> 12)):
                    #    continue
                    temp_file =  str(sys.argv[1]) + "\\downsize_jpeg_encode\\" + "reso_" + str(reso_list[i]) + "_jpeg_encode_" + str(int(qp_list[j]))
                    #print("\n ===========\n PNG Downsize %d + JPEG + Encode (QP) %d (in KB) \n ============ \n" %(reso_list[i],qp_list[j]))
                    mech_list.append("\\downsize_jpeg_encode\\reso_" + str(reso_list[i]) + "_jpeg_encode_" + str(int(qp_list[j])))
                    sz_list.append(size(temp_file, 2, x))
            #JPEG+Encode
            for j in range(len(qp_list)):
                temp_file =  str(sys.argv[1]) + "\\jpeg_encode\\" + "jpeg_encode_" + str(int(qp_list[j]))
                #print("\n ===========\n JPEG + Encode (QP) %d (in KB) \n ============ \n" %(qp_list[j]))
                mech_list.append("\\jpeg_encode\\jpeg_encode_" + str(int(qp_list[j])))
                sz_list.append(size(temp_file,3, x))
            #"""
            
            #"""
            #used for parallel execution only
            #print(len(list2))
            print("\n =================== \n configurations possible under %d Mbps BW \n ======================== \n" %ub)
            #bw_str = "bw_0417_" + str(ub)
            #os.system("mkdir " + dir_name + bw_str)
            count(sz_list, mech_list, lb*scale, ub*scale, temp_fol, bw_str, x)#x--> frame_id
            #res = [idx for idx, val in enumerate(sz_list) if val < ub*scale]
            #print("\n possible configurations =" + str(len(res)))
            #print(len(temp))
            """
            
            bw_list = [1,2,5,10,20,30,40,60]#,80]
            #scale = 8.75
            scale = 10.25
            #this part is used while plotting graphs
            for i in range(len(bw_list)):
                print("\n =================== \n configurations possible under %d Mbps BW \n ======================== \n" %bw_list[i])
                bw_str = "bw_" + str(bw_list[i])
                #os.system("mkdir bbox_drone\\" + bw_str)
                ub = bw_list[i]
                if(i==0):
                    lb = 0
                else:
                    lb = bw_list[i-1]
                temp = count(sz_list, lb*scale, ub*scale, temp_fol, bw_str)
                #temp = count(sz_list, (bw_list[i]*scale)/1.2, (bw_list[i]*scale)*1.14, temp_fol, bw_str)
                print(temp)
            """
            
            
        
        
                
if __name__ == "__main__":
	main()

				
				
		
				
				
			



