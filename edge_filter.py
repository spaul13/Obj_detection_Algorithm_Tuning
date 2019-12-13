import cv2
import time
from scipy import ndimage, misc
image_dir = '0319_objdetection_expr/test_5/'
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image

temp = "test_5_Laplace/"
percept_list = []

log_list = []
canny_list = []
img_dir = "H:\\drone_video_cp_7\\pic_"

def plot(data, title):
    plot.i += 1
    plt.subplot(2,2,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
plot.i = 0

def Gaussian_HP(image_path, sigma, output_path):#High pass filter using gaussian LPF
    img = Image.open(image_path)
    data = np.array(img, dtype= float)
    lowpass = ndimage.gaussian_filter(data,sigma)
    gauss_highpass = data - lowpass
    misc.imsave(output_path, gauss_highpass)
    return gauss_highpass

def LoG(image_path, sigma, output_path):#Laplacian of Gaussian filter 
    img = Image.open(image_path)
    data = np.array(img, dtype= float)
    log_highpass = ndimage.gaussian_laplace(data,sigma)
    misc.imsave(output_path, log_highpass)
    return log_highpass

def laplacehp(image_path, output_path):#kernel size not mentioned
    img = cv2.imread(image_path,0)
    laplacian = cv2.Laplacian(img,cv2.CV_64F) #depth size
    cv2.imwrite(output_path, laplacian)
    return laplacian
    
def canny_edgedetector(image_path, output_path):
    img = cv2.imread(image_path,0)
    edges = cv2.Canny(img,100,200)
    cv2.imwrite(output_path, edges)
    return edges

def main():
    """
    image_path = "F:\decoded_jockey\pic_1.png"
    output_path = "F:\decoded_jockey\pic_1_edges_canny.png"
    temp = canny_edgedetector(image_path, output_path)
    """
    for i in range(1, 421): #for 3 need to do (251,265)
        print(i)
        image_path = img_dir + str(i) + "_org.png"
        output_path = img_dir + str(i) + "_org_log.png"
        temp = LoG(image_path, 3, output_path)
        stat_info = os.stat(image_path)
        stat_info_out = os.stat(output_path)
        norm_size = stat_info_out.st_size / (stat_info.st_size)
        print(norm_size)
        log_list.append(norm_size)
        #os.system("rm " + output_path)
        output_path = img_dir + str(i) + "_org_canny.png"
        temp = canny_edgedetector(image_path,output_path)
        #stat_info = os.stat(image_path)
        stat_info_out = os.stat(output_path)
        norm_size = stat_info_out.st_size / (stat_info.st_size)
        canny_list.append(norm_size)
        print(norm_size)
        #os.system("rm " + output_path)
    print("\n ========= \n " + img_dir + "\n ====== \n")    
    print("\n === After canny edge detector, normalized size === \n")
    print(canny_list)
    print(str(max(canny_list)) + "," + str(min(canny_list)))
    
    print("\n === After LoG edge detector, normalized size === \n")
    print(log_list)
    print(str(max(log_list)) + "," + str(min(log_list)))
    
        
    """
    output_path = "F:\decoded_jockey\pic_1_edges_laplacehp.png"
    temp = laplacehp(image_path, output_path) # laplaceHP only producing black frames (not working)
    output_path = "F:\decoded_jockey\pic_1_edges_LoG.png"
    temp = LoG(image_path, 3, output_path)
    output_path = "F:\decoded_jockey\pic_1_edges_GHP.png"
    temp = Gaussian_HP(image_path, 3, output_path)
    """
    
    

if __name__ == '__main__':
    main()

                



















"""
    q=os.listdir(image_dir)
    for j in q:
        image_path = image_dir + j
        fid = str((j.split('.'))[0])
        print [fid]
        temp_file = temp + fid + '.png'
        #for laplacian only
        laplacehp(image_path, temp_file)
        
        img = Image.open(image_path)
        data = np.array(img, dtype= float)
        #plot(data, 'original')
        dst = LoG(data, 3)
        #print [time.time()*1000 - prev_time]
        misc.imsave(temp_file, dst)
        #plot(dst,'gaussian')
        #dst = LoG(data,3)
        #plot(dst,'LoG')

        plt.show()
"""
   
