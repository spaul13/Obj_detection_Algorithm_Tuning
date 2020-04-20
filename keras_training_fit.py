import cv2, os, random, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras
import tensorflow as tf
import os.path, statistics
import calculate_iou_2obj as cal_mul
import glob
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split


train_dir = "C:\\yolo\\darknet\\build\\darknet\\x64\\data\\obj_0204_prev\\123\\"
train_pics = os.listdir(train_dir)
#its important to generate a random seed and shuffle
#random.seed(40)
#random.shuffle(train_pics)

print(train_pics)
#define the image size
nrows, ncols, channels = 416, 416, 3

def read_and_process_image(list_imgs):
    X, Y = [], []
    for i in range(len(list_imgs)):
        X.append(cv2.resize(cv2.imread(train_dir + list_imgs[i], cv2.IMREAD_COLOR), (nrows, ncols), interpolation=cv2.INTER_CUBIC)) #Read the Image
        """
        image = cv2.imread(list_imgs[i])
        image = cv2.resize(image, (nrows, ncols))
        image = img_to_array(image)
        X.append(image)
        """
        Y.append(0)

    return np.array(X),np.array(Y) #array of image pixel values need to be converted to numpy for training

train_X, train_Y = read_and_process_image(train_pics)
print("Shape of train image is:", train_X.shape)
#later can use train_test_split for splitting data for training and validation

def find_acc_loss(orgfile, testfile):
    cmd_str = "python detect.py --images " + orgfile + " --reso 1024 --file_name log_org.txt"
    os.system(cmd_str)
    cmd_str = "python detect.py --images " + testfile + " --reso 1024 --file_name log_test.txt"
    os.system(cmd_str)
    acc_loss = cal_mul.cal_iou("log_org.txt", "log_test.txt")
    return acc_loss


def check_size(outfile):
    statinfo = os.stat(outfile)
    if(statinfo.st_size > tx_size):
        return -1 #add large penalty
    else:
        return 0
    


def compress(infile, reso, jpeg_qp, encoding_qp):
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
    cmd_str = "ffmpeg -i temp_1.png -vf scale=" + str(org_res) +":-1 temp_2.png"
    os.system(cmd_str)
    acc_loss = find_acc_loss(infile, "temp_2.png")
    bool_sz = check_size("temp_2.png")
    #deleting
    os.system("rm temp*")
    """
    os.system("rm temp.jpg")
    os.system("rm temp.mp4")
    os.system("rm temp_1.png")
    os.system("rm temp_2.png")
    """
    return acc_loss, bool_sz
    
	
	

def customized_loss(y_actual, y_predicted):
	#y_pred = [0.4, 0.5, 0.1]
    batch_loss = 0.0
    print("\n .... Going in Customized loss .... \n")
    for i in train_X:
        print(len(i))
        idx = (train_generator.batch_index - 1) * train_generator.batch_size
        print(idx)
        print(train_generator.filenames[idx : idx + train_generator.batch_size]) 
        for j in range(idx, idx + train_generator.batch_size):
            print("\n ....... before predict_generator ..... \n")
            print(train_generator)
            print("After 1 \n")
            print(train_generator[j])
            print("After 2 \n")
            out_pred = model.predict_generator(train_generator,1)#error here
            print("\n ....... after predict_generator ..... \n")
            print(np.shape(out_pred))
            y_pred = out_pred[j]
            print(out_pred)
            print(y_pred)
            reso = int(y_pred[0]*org_res)
            jpeg_qp = int(best_jpeg/y_pred[1])
            encoding_qp = int(best_qp/y_pred[2])
            print(y_pred[0], y_pred[1], y_pred[2])
            print(reso, jpeg_qp, encoding_qp)
            pic_org = inp_dir + train_generator.filenames[j]
            acc_loss, bool_sz = compress(pic_org, reso, jpeg_qp, encoding_qp)
            print(acc_loss, bool_sz)
            batch_loss += acc_loss + bool_sz * file_size_mismatch_loss
    print("\n .... Going Out Customized loss .... \n")
    return batch_loss/train_generator.batch_size

#define the yolov3-tiny model
def define_train_model():
    model.add(Conv2D(16, (3, 3), input_shape=(416, 416, 3)))#(3, 416, 416)
    model.add(Activation('relu'))
    #"""
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #to avoid overfitting
    model.add(Flatten())
    model.add(Dropout(0.5)) #reduce the size of the network
    #"""
    #Regression layer
    model.add(Dense(num_compression, activation='sigmoid'))
    model.summary()
    
    model.compile(loss=customized_loss, optimizer='rmsprop')
    
    model.fit(train_X, train_Y, steps_per_epoch=100 // batch_size, epochs=50)



