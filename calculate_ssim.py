from skimage.measure import compare_ssim
import argparse
import imutils
import cv2, statistics, sys, os

qp_list = [22, 25, 28, 36]
reso_list = [384, 544, 672, 768]
jpeg_list = [1440, 1920, 3840]
freq_list = [2, 3, 4, 6, 8]
MAX_NUM = 400
ssim_list = []



def encode_ssim(infol):
    temp_fol = infol
    for i in range(len(qp_list)):
        ssim_list = []
        print("\n==========\n Encode %d \n=============\n" %qp_list[i])
        for j in range(1, MAX_NUM+1):
            infol = temp_fol
            path1 = infol + "\\raw_frames\\pic_" + str(j) +"_org.png"
            infol += "\\" + str(reso_list[i])
            path2 = infol + "\\pic_" + str(j)+"_"+str(qp_list[i])+".mp4"
            cap2 = cv2.VideoCapture(path2)
            ret2, imageB = cap2.read()
            imageA = cv2.imread(path1)
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            # 5. Compute the Structural Similarity Index (SSIM) between the two
            (score, diff) = compare_ssim(grayA, grayB, full=True)
            #diff = (diff * 255).astype("uint8")
            #print(score)
            ssim_list.append(score)
        print(ssim_list)
        print("\n Avg: " + str(statistics.mean(ssim_list)))
        print("\n stddev: " + str(statistics.stdev(ssim_list)))



def downsize_ssim(infol):
    temp_fol = infol
    for i in range(len(qp_list)):
        ssim_list = []
        print("\n==========\n Downsize %d \n=============\n" %reso_list[i])
        for j in range(1, MAX_NUM+1):
            infol = temp_fol
            path1 = infol + "\\raw_frames\\pic_" + str(j) +"_org.png"
            infol += "\\" + str(reso_list[i])
            path2 = infol + "\\pic_" + str(j)+"_"+str(reso_list[i])+"_up.png"
            imageB = cv2.imread(path2)
            imageA = cv2.imread(path1)
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            # 5. Compute the Structural Similarity Index (SSIM) between the two
            (score, diff) = compare_ssim(grayA, grayB, full=True)
            #diff = (diff * 255).astype("uint8")
            #print(score)
            ssim_list.append(score)
        print(ssim_list)
        print("\n Avg: " + str(statistics.mean(ssim_list)))
        print("\n stddev: " + str(statistics.stdev(ssim_list)))            

def jpeg_ssim(infol):
    temp_fol = infol
    for i in range(len(jpeg_list)):
        ssim_list = []
        print("\n==========\n JPEG Downsize %d \n=============\n" %jpeg_list[i])
        for j in range(1, MAX_NUM+1):
            infol = temp_fol
            path1 = infol + "\\raw_frames\\pic_" + str(j) +"_org.png"
            infol += "\\" + str(jpeg_list[i])
            path2 = infol + "\\pic_" + str(j)+"_"+str(jpeg_list[i])+"_up.jpg"
            imageB = cv2.imread(path2)
            imageA = cv2.imread(path1)
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            # 5. Compute the Structural Similarity Index (SSIM) between the two
            (score, diff) = compare_ssim(grayA, grayB, full=True)
            #diff = (diff * 255).astype("uint8")
            #print(score)
            ssim_list.append(score)
        print(ssim_list)
        print("\n Avg: " + str(statistics.mean(ssim_list)))
        print("\n stddev: " + str(statistics.stdev(ssim_list)))


def freq_ssim(infol):
    temp_fol = infol
    for i in range(len(freq_list)):
        ssim_list = []
        print("\n==========\n Freq reduction %d \n=============\n" %freq_list[i])
        for j in range(1, MAX_NUM+1):
            infol = temp_fol
            path1 = infol + "\\raw_frames\\pic_" + str(j) +"_org.png"
            if(j%freq_list[i]==1):
                path2 = infol + "\\raw_frames\\pic_" + str(j) +"_org.png"
            imageB = cv2.imread(path2)
            imageA = cv2.imread(path1)
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            # 5. Compute the Structural Similarity Index (SSIM) between the two
            (score, diff) = compare_ssim(grayA, grayB, full=True)
            #diff = (diff * 255).astype("uint8")
            #print(score)
            ssim_list.append(score)
        print(ssim_list)
        print("\n Avg: " + str(statistics.mean(ssim_list)))
        print("\n stddev: " + str(statistics.stdev(ssim_list)))
        

def main():
    print(len(sys.argv))
    if len(sys.argv) is not 2:
        print("Usage: program image_directory")
    else:
        infol = sys.argv[1]
        print(str(len(sys.argv)) + "," + str(infol))
        encode_ssim(infol)
        downsize_ssim(infol)
        jpeg_ssim(infol)
        freq_ssim(infol)

if __name__ == "__main__":
	main()

















"""   
counter=0    
while(cap1.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap1.read()
  if ret == True:
    print("True")
    counter+=1
    
print("\n current counter = %d" %counter)
"""