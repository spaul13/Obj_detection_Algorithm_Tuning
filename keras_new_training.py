# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import keras
import tensorflow as tf
#import keras.backend.tensorflow_backend as K
import os.path
import os, statistics
import calculate_iou_2obj as cal_mul
import glob
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


#for training and test data generation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

org_res_train = 3840
org_res = 3840
best_jpeg = 10
best_qp = 10
bw = 5
comm_time = 50
tx_size = (bw * 1024 * comm_time)/8000
model = Sequential()
num_compression = 3
batch_size = 1
inp_dir = "C:\\yolo\\darknet\\build\\darknet\\x64\\data\\obj_0204_prev\\"
#inp_dir = "H:\\train\\train\\"
file_size_mismatch_loss = 1000 #needs to very high if file size > threshold

train_datagen = ImageDataGenerator(rotation_range=100, shear_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
train_datagen = ImageDataGenerator(rescale=1./255)
# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

train_generator = train_datagen.flow_from_directory(inp_dir,target_size=(416,416),  # all images will be resized to 150x150
        batch_size=batch_size)#,class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
train_generator = train_datagen.flow_from_directory(inp_dir,target_size=(416,416),batch_size=batch_size, class_mode=None, shuffle=False)

#A `DirectoryIterator` yielding tuples of `(x, y)`where `x` is a NumPy array containing a batch
#of images with shape `(batch_size, *target_size, channels)`and `y` is a NumPy array of corresponding labels.

print(len(train_generator))
"""
counter = 0
for i in train_generator:
    print(np.shape(i),np.size(i))
    counter+=1
print("\n total counter value = %d\n"%(counter))
"""


#get the filenames of the train_generator
"""
for i in train_generator:
    print(len(i))
    idx = (train_generator.batch_index - 1) * train_generator.batch_size
    #print(idx)
    print(train_generator.filenames[idx : idx + train_generator.batch_size])
"""    


#need to write customized dataloader

def find_acc_loss(orgfile, testfile):
    cmd_str = "python detect.py --images " + orgfile + " --reso 1024 --file_name log_org"
    os.system(cmd_str)
    cmd_str = "python detect.py --images " + testfile + " --reso 1024 --file_name log_test"
    os.system(cmd_str)
    #statistics.mean(bbox_accuracy), mismatch_org, total_org, total, correct_detection
    acc_loss, mismatch_org, total_org, total, correct_detection = cal_mul.cal_iou('log_org.txt', 'log_test.txt')
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
    cmd_str = "ffmpeg -i temp.mp4 temp_1.jpg"
    os.system(cmd_str)
    cmd_str = "ffmpeg -i temp_1.jpg -vf scale=" + str(org_res) +":-1 temp_1_org.jpg"
    os.system(cmd_str)
    acc_loss = find_acc_loss(infile, "temp_1_org.png")
    bool_sz = check_size("temp_1_org.jpg")
    #deleting
    os.system("rm temp*")
    """
    os.system("rm temp.jpg")
    os.system("rm temp.mp4")
    os.system("rm temp_1.png")
    os.system("rm temp_1_org.png")
    """
    return acc_loss, bool_sz
    
	
	

def customized_loss(y_actual, y_predicted):
	#y_pred = [0.4, 0.5, 0.1]
    batch_loss = 0.0
    print("\n .... Going in Customized loss .... \n")
    for i in train_generator:
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
            y_pred = out_pred[0] #not out_pred[j]
            print("SP: Shape of out_pred is:",np.shape(out_pred))
            print(out_pred)
            print("SP: Shape of y_pred is:",np.shape(y_pred))
            print(y_pred)
            reso = int(y_pred[0]*org_res_train)
            jpeg_qp = int(best_jpeg/y_pred[1])
            encoding_qp = int(best_qp/y_pred[2])
            pic_org = inp_dir + train_generator.filenames[j]
            acc_loss, bool_sz = compress(pic_org, reso, jpeg_qp, encoding_qp)
            print(acc_loss, bool_sz)
            batch_loss += (1-acc_loss)*100 + bool_sz * file_size_mismatch_loss
            print("\n SP: original file name = %s, acc loss = %f, size diff = %d, total_loss = %f \n" %(pic_org, acc_loss, bool_sz, batch_loss))
            print(reso,jpeg_qp,encoding_qp)
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
    #before the training check the model predictions
    print("\n predict_generator output before training \n")
    predict_generator_output = model.predict_generator(train_generator,1)
    print("\n outputing predict generator for each element \n")
    print(np.shape(predict_generator_output), np.size(predict_generator_output))
    counter=0
    for j in predict_generator_output:
        print(j)
        counter+=1
    print("\n total counter value = %d\n"%counter)
    
    #start training
    print("\n .... before started training .... \n")
    model.compile(loss=customized_loss, optimizer='rmsprop')
    #"""
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    
    model.fit_generator(train_generator, steps_per_epoch=100 // batch_size, epochs=50)
    print(model.predict_generator(train_generator,1))
    #"""

def customized_dataloader():
    file_list = glob.glob(inp_dir + "*.png")
    print(file_list)
    label_list =[1 for i in range(len(file_list))]
    

def main():
    print("Main\n")
    #customized_dataloader()
    define_train_model()


if __name__ == "__main__":
	main()   
    
    
    
	