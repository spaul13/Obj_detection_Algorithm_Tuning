import numpy as np
import cv2
from matplotlib import pyplot as plt
import statistics


fol_list = [5,6,7]#[3, 4, 5]#[2]
#max_list = [1]#, 1, 1, 1]#, 1]
max_list = [144, 350, 420]#[264, 216, 144]#, 350]#[250]

pic_prefix = "E:\\drone_video_cp_"
pic_prefix = "H:\\drone_video_cp_"


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
    print("\n mean dist = %f, stdev = %f, matched ratio = %f" %(statistics.mean(dist) , statistics.stdev(dist), good_ratio))
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
    print("\n mean dist = %f, stdev = %f, matched ratio = %f" %(statistics.mean(dist) , statistics.stdev(dist), good_ratio))
    return statistics.mean(dist) , statistics.stdev(dist), good_ratio
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None,flags=2)
    #plt.imshow(img3),plt.show()


def main():
    for i in range(len(fol_list)):
        pic_new_prefix = pic_prefix + str(fol_list[i])+"//"
        mean_dist_list, std_dist_list, good_ratio_list = [], [], []
        for j in range(1, max_list[i]):
            file1 = pic_new_prefix +  "pic_" + str(j+1) +"_org.png"
            file2 = pic_new_prefix +  "pic_" + str(j) +"_org.png"
            print("\n ===== frame no = %d" %j)
            mean_dist, std_dist, good_ratio = SIFT_features(file1, file2)
            #mean_dist, std_dist, good_ratio = ORB_features(file1, file2)
            mean_dist_list.append(mean_dist)
            std_dist_list.append(std_dist)
            good_ratio_list.append(good_ratio)
            
            #ORB_features(file1, file2)
        #print("\n ==== orb ====== \n")
        print("\n ==== sift ====== \n")
        print("\n ====== %d ======= \n" %fol_list[i])
        print("\n ------ avg dist ------ \n")
        print(mean_dist_list)
        print("\n ------ dist standard deviation ------ \n")
        print(std_dist_list)
        print("\n ------- number of almost matched feature points ------- \n")
        print(good_ratio_list)
        
if __name__ == "__main__":
	main()	
    