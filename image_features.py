from PIL import Image, ImageStat
import math
import numpy as np
import cv2
#import imutils

#pic_prefix = "E://drone_video_cp_"
pic_prefix = "H://drone_video_cp_"
MAX_NUM = 4
fol_list = [7] #[2, 3, 4, 5, 6, 7]
#max_list = [1, 1, 1, 1]#, 1]
max_list = [420] #[250, 264, 216, 144, 350, 420]




def colorfulness(im_file):
    #image = cv2.imread(im_file)
    image = Image.open(im_file)
    #image = imutils.resize(image, width=250)
    #(B, G, R) = cv2.split(image.astype("float"))
    for R, G, B in image.getdata():
        #R, G, B = image.getdata()
        # compute rg = R - G
        #print(str(R) + "," +str(G) + "," +str(B))
        rg = np.absolute(R - G)
        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)
        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)


def brightness( im_file ):
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)) 
         for r,g,b in im.getdata())
   return sum(gs)/stat.count[0]

def cal_contrast(im_file):
    img = cv2.imread(im_file)
    """
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
    # compute min and max of Y
    min = np.min(Y)
    max = np.max(Y)
    # compute contrast
    contrast = (max-min)/(max+min)
    print(min,max,contrast)
    """
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #RMS contrast
    contrast = img_grey.std()
    #print(contrast)
    return contrast
    #"""
   
def main():
    for i in range(len(fol_list)):
        pic_new_prefix = pic_prefix + str(fol_list[i])+"//"
        bright_list = []
        for j in range(1, max_list[i]+1):
            pic_name = pic_new_prefix +  "pic_" + str(j) +"_org.png"
            #print(pic_name)
            bright_list.append(brightness(pic_name))
            #bright_list.append(colorfulness(pic_name))
            #bright_list.append(cal_contrast(pic_name))
        
        print("\n ====== Brightness for Drone Video %d ======= \n" %fol_list[i])
        print(bright_list)
            
		

if __name__ == "__main__":
	main()	
	