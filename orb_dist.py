import numpy as np
import cv2
from matplotlib import pyplot as plt
import statistics


fol_list = [5,6,7]#[3, 4, 5]#[2]
#max_list = [1]#, 1, 1, 1]#, 1]
max_list = [144, 350, 420]#[264, 216, 144]#, 350]#[250]

pic_prefix = "E:\\drone_video_cp_"
pic_prefix = "H:\\drone_video_cp_"

total_frame_number = [249, 263, 419, 440, 299, 419, 143, 239, 329, 359, 360, 360, 261, 420, 288, 121, 170, 325]
total_frame_number.extend([215, 143, 349, 229, 419, 143, 299, 390, 263, 199, 184, 300])
dir  = [2,3,7,9,11,12,15,16,20,21,22, 24,26,27,29,30,32,34]
dir.extend([4, 5, 6, 10, 13, 17, 19, 23, 25, 28, 31, 33])
dir_sort = sorted(dir)
total_sort = []

for i in range(len(dir_sort)):
    total_sort.append(total_frame_number[dir.index(dir_sort[i])])
total_frame_number = total_sort
dir = dir_sort

#extended partition
dir.extend([35, 36, 37, 38])
total_frame_number.extend([384,145,180,210])


def ORB_features(file1, file2):
    img1 = cv2.imread(file1,0)
    img2 = cv2.imread(file2,0) # trainImage
    orb = cv2.ORB_create()        # Initiate SIFT detector
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    dist = []
    for m in matches:
        #print("\n current distance is = " + str(m.distance))
        dist.append(m.distance)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # for K>1, crosscheck needs to be false
    match = bf.knnMatch(des1,des2, k=2)
    dist_m = []
    good = []
    for m,n in match:
        #print("\n current distance = %f" %(m.distance))
        dist_m.append(m.distance)
        if m.distance < 0.75*n.distance:
            #print("\n current one is good")
            good.append([m])
    if(len(dist_m) > 0):
        good_ratio = len(good)/len(dist_m)
    else:
        good_ratio = 0.0
    #print("\n mean dist = %f, stdev = %f, matched ratio = %f" %(statistics.mean(dist) , statistics.stdev(dist), good_ratio))
    return statistics.mean(dist) , statistics.stdev(dist), good_ratio
    # Draw first 10 matches.
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)
    #plt.imshow(img3)
    #plt.show()

def SIFT_features(file1, file2):
    img1 = cv2.imread(file1,0)
    img2 = cv2.imread(file2,0)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    match = bf.match(des1,des2)
    dist = []
    for m in match:
        dist.append(m.distance)
        #print("\n only distance is = " + str(m.distance))
    # Apply ratio test
    dist_m = []
    good = []
    for m,n in matches:
        #print("\n current distance = %f" %(m.distance))
        dist_m.append(m.distance)
        if m.distance < 0.75*n.distance:
            #print("\n current one is good")
            good.append([m])
    
    #print(str(len(dist)) + "," + str(len(good)) + "," + str(len(dist_m)))
    good_ratio = len(good)/len(dist_m)
    #print("\n mean dist = %f, stdev = %f, matched ratio = %f" %(statistics.mean(dist) , statistics.stdev(dist), good_ratio))
    return statistics.mean(dist) , statistics.stdev(dist), good_ratio
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None,flags=2)
    #plt.imshow(img3),plt.show()


def main():
    for i in range(len(dir)):
        pic_new_prefix = pic_prefix + str(dir[i])+"\\"
        mean_dist_list, std_dist_list, good_ratio_list = [], [], []
        mean_dist_list_orb, std_dist_list_orb, good_ratio_list_orb = [], [], []
        for j in range(1, total_frame_number[i]+1):
            file1 = pic_new_prefix +  "pic_" + str(j+1) +"_org.png"
            file2 = pic_new_prefix +  "pic_" + str(j) +"_org.png"
            #print("\n ===== frame no = %d" %j)
            mean_dist, std_dist, good_ratio = SIFT_features(file1, file2)
            mean_dist_orb, std_dist_orb, good_ratio_orb = ORB_features(file1, file2)
            mean_dist_list.append(mean_dist)
            std_dist_list.append(std_dist)
            good_ratio_list.append(good_ratio)
            mean_dist_list_orb.append(mean_dist)
            std_dist_list_orb.append(std_dist)
            good_ratio_list_orb.append(good_ratio)
            
        print("\n ==== sift_%d ====== \n"%dir[i])
        print("\n ------ avg dist ------ \n")
        print(mean_dist_list)
        print("\n ------ dist standard deviation ------ \n")
        print(std_dist_list)
        print("\n ------- number of almost matched feature points ------- \n")
        print(good_ratio_list)
        print("\n ==== ORB_%d ====== \n"%dir[i])
        
        print("\n ------ avg dist ------ \n")
        print(mean_dist_list_orb)
        print("\n ------ dist standard deviation ------ \n")
        print(std_dist_list_orb)
        print("\n ------- number of almost matched feature points ------- \n")
        print(good_ratio_list_orb)
        
if __name__ == "__main__":
	main()	
    