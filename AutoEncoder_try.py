import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np
import PCA_image as pi
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_val = x_test[:7000]
x_test = x_test[7000:]
print("validation data: {0} \ntest data: {1}".format(x_val.shape, x_test.shape))
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#input_img = Input(shape=(32, 32, 3))

train_xlist = ['H:\\drone_video_cp_5\\pic_1_org.png', 'H:\\drone_video_cp_5\\pic_2_org.png', 'H:\\drone_video_cp_5\\pic_3_org.png', 'H:\\drone_video_cp_5\\pic_4_org.png', 'H:\\drone_video_cp_5\\pic_5_org.png', 'H:\\drone_video_cp_5\\pic_6_org.png']#, 'H:\\drone_video_cp_5\\pic_10_org.png', 'H:\\drone_video_cp_5\\pic_11_org.png', 'H:\\drone_video_cp_5\\pic_12_org.png']
x_train_list = []
for ele in train_xlist:
    img = Image.open(ele)
    data = img_to_array(img)
    #data = np.array(img, dtype= float)
    x_train_list.append(data)
x_train =  np.asarray(x_train_list) 
print(x_train.shape)

test_xlist = ['H:\\drone_video_cp_5\\pic_7_org.png', 'H:\\drone_video_cp_5\\pic_8_org.png', 'H:\\drone_video_cp_5\\pic_9_org.png']
x_test_list = []
for ele in test_xlist:
    img = Image.open(ele)
    data = img_to_array(img)
    #data = np.array(img, dtype= float)
    x_test_list.append(data)
x_test =  np.asarray(x_test_list) 
print(x_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape, x_test.shape)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape, x_test.shape)


encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
print(x_train.shape[-1])
# this is our input placeholder
input_img = Input(shape=(x_train.shape[-1],))#784
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(x_train.shape[-1], activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))



"""
train_xlist = ['H:\\drone_video_cp_5\\pic_1_org.png', 'H:\\drone_video_cp_5\\pic_2_org.png', 'H:\\drone_video_cp_5\\pic_3_org.png', 'H:\\drone_video_cp_5\\pic_4_org.png', 'H:\\drone_video_cp_5\\pic_5_org.png', 'H:\\drone_video_cp_5\\pic_6_org.png', 'H:\\drone_video_cp_5\\pic_7_org.png', 'H:\\drone_video_cp_5\\pic_8_org.png', 'H:\\drone_video_cp_5\\pic_9_org.png']#, 'H:\\drone_video_cp_5\\pic_10_org.png', 'H:\\drone_video_cp_5\\pic_11_org.png', 'H:\\drone_video_cp_5\\pic_12_org.png']
x_train_list = []
for ele in train_xlist:
    img = Image.open(ele)
    data = img_to_array(img).astype("float32")/255.
    #data = np.array(img, dtype= float)
    x_train_list.append(data)
    print(data.shape)
    print(len(data))
x_train =  np.asarray(x_train_list) 

print(x_train[0].shape)   

shape_img = x_train[0].shape
shape_img_flattened = (np.prod(list(shape_img)),)
input_img = Input(shape=shape_img_flattened)
print(1)
print(input_img.shape)
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
#decoded = Dense(784, activation='sigmoid')(encoded)
print(2)
print(encoded.shape)
# this model maps an input to its reconstruction
#autoencoder = Model(input_img, decoded)
print(3)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
print(4)
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(5)
encoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True)
                #validation_data=(x_test, x_test))
print("End")
"""