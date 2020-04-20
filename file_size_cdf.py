import numpy as np
import scipy, os
import matplotlib.pyplot as plt
import seaborn as sns
import math, cv2
import sys
import statistics

checked_fol_list = [2,3,10,11,17,22,24,25,27,29,31,35,38] 
length_list = [249, 263, 229, 299, 143, 360, 360, 263, 420, 288, 184, 384, 210]
checked_fol_list.extend([4,5,15,21,23,26,28,30,32,36,37])
length_list.extend([215, 143, 143, 359, 390, 261, 199, 121, 170, 145, 180])
file_size = []
low_fol, mid_fol, high_fol = [], [], []
low_file, mid_file, high_file = [], [], []
#file_size = [1,2,3]
#"""
#print(sys.argv[1])
#list_index = int(sys.argv[1])
for list_index in range(len(checked_fol_list)):
    file_size = []
    for i in range(list_index, list_index+1):
        fol_name = "H:\\drone_video_cp_" + str(checked_fol_list[i])
        print("\n ======== \n %s \n ========== \n" %fol_name)
        for j in range(1, length_list[i]+1):
            file_path = fol_name + "\\pic_" + str(j) + "_org.png"
            """
            #this is to check the resolution
            if(j==1):
                im = cv2.imread(file_path)
                h, w, c = im.shape
                print(w,h)
                if((w!=3840)):
                    print("Not 4K\n")
            """                
            stat_info = os.stat(file_path)
            # get size of file in bytes
            size = (stat_info.st_size)/(1024*1024)
            """
            #low complexity
            if(size<=3):#10 percentile
                print("\n low complexity: fol = %d, file = %d" %(checked_fol_list[i],j))
                low_fol.append(checked_fol_list[i])
                low_file.append(j)
            #medium complexity
            if((size>8) and (size<=11)):#50 percentile
                print("\n medium complexity: fol = %d, file = %d" %(checked_fol_list[i],j))
                mid_fol.append(checked_fol_list[i])
                mid_file.append(j)
            if((size>15)):#over 90 percentile
                print("\n high complexity: fol = %d, file = %d" %(checked_fol_list[i],j))
                high_fol.append(checked_fol_list[i])
                high_file.append(j)
            """
            """
            if(size<=0.4):
                print("\n the small file size index is = " , j)
            if(size>19.5):
                print("\n the large file size index is = " , j
            """
            file_size.append(size)
    """
    print("\n ============ \n Low complexity \n ============ \n")
    print(low_fol)
    print(low_file)
    print("\n ============ \n Medium complexity \n ============ \n")
    print(mid_fol)
    print(mid_file)
    print("\n ============ \n high complexity \n ============ \n")
    print(high_fol)
    print(high_file)
    """



    #cdf_pts = list(range(1, math.ceil(max(file_size))+1,1))
    cdf_pts = list(np.linspace(1, math.ceil(max(file_size)),20))
    cdf_list = [0.0]*len(cdf_pts)
    for i in range(len(cdf_pts)):
        count=0
        for j in range(len(file_size)):
            if(file_size[j]<=cdf_pts[i]):
                count+=1
        cdf_list[i] = (count)/len(file_size)

    print(cdf_pts)
    print(cdf_list)        

    #calculate the CDF
    #norm_cdf = scipy.stats.norm.cdf(file_size)
    #print(norm_cdf)
    print(min(file_size), max(file_size), len(file_size), len(cdf_list))
    print("\n minimum file size = %f MB " %min(file_size))
    print("\n maximum file size = %f MB" %max(file_size))
    print("\n standard deviation = %f MB" %statistics.mean(file_size))
    print("\n standard deviation = %f MB" %statistics.stdev(file_size))
    # plot the cdf
    """
    plt.figure()
    plt.xlabel("File Size (in MB)", fontsize = 30)
    plt.ylabel("CDF", fontsize = 30)
    cdf_plt_pts = list(np.linspace(1, math.ceil(max(file_size)),8))
    plt.xticks(cdf_plt_pts, fontsize = 12)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1.0], fontsize = 25)
    sns.lineplot(x=cdf_pts, y=cdf_list)
    print("\n the current folder index = %d \n" %checked_fol_list[list_index])
    title = str(checked_fol_list[list_index]) + "min:"+str(min(file_size))+", max:" + str(max(file_size)) + ", stdev:" + str(statistics.stdev(file_size))
    plt.title(title)
    #plt.show()
    save_file = "file_size_fol\\file_size_cdf_" + str(checked_fol_list[list_index]) + ".png"
    plt.savefig(save_file, bbox_inches='tight')
    """

		
		