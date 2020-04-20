#this code is used to log the groundtruth best configuration

import numpy as np
import matplotlib.pyplot as plt
import statistics, os, sys
import calculate_iou_2obj as cal_mul
from matplotlib.cm import get_cmap
import operator
#reso_list = [3200, 2560, 2048, 1440, 960, 768, 640, 480, 320, 160]
reso_list = [2048, 1440, 960, 768, 640, 480, 320, 160]
#qp_list = np.linspace(0,40,11)
qp_list = np.linspace(12,40,8)
jpeg_enabled = True
org_res = 3840

#all configurations until bw = 5
bw_list = [1,2,5,10,20,30,40,60]#,80]
bw = 5
#configurations allowable between specified and previous here 1-2 Mbps
#bw_list = [2,5,10,20,30,40,60]
#bw = 2 # to reduce number of configs

prefix = "bw_0417_" #bw_tiny_, bw_tiny_416


new_list = []
complete_config_list = []

def main():
    print(len(sys.argv))
    config_list = []
    if (len(sys.argv) < 3):
        print("Usage: python program_name directory max_image_number")
    else:
        temp_fol = sys.argv[1]
        MAX_NUM = int(sys.argv[2])
        print(str(len(sys.argv)) + "," + str(temp_fol))
    for i in range(len(bw_list)):
        if(bw_list[i]>bw):
            break
        else:
            file_list = os.listdir(temp_fol + "\\" +prefix +str(bw_list[i]) +"\\")
            new_file_list = [prefix +str(bw_list[i]) +"\\" + fl for fl in file_list]
            complete_config_list.extend(new_file_list)
            print(len(file_list))
            
    print(len(complete_config_list))
    best_config_list, best_acc_list = [], []
    for i in range(1, MAX_NUM+1):
        substring = "_" + str(i) + ".txt"
        frame_config_list, iou_list, config_list = [], [], []
        for ccl in complete_config_list:
            if substring in ccl:
                frame_config_list.append(ccl)
        print("\n # of configurations for frameid %d is %d" %(i,len(frame_config_list)))
        orgfile = temp_fol + "\\bbox_org\\" +  "log_bbox_org_" +  str(i) +".txt"
        for fcl in frame_config_list:
            config = (fcl.split("\\")[-1]).replace(substring,'')
            config_list.append(config)
            sampled_file = temp_fol + "\\" + fcl
            iou, mismatch_org, predorg, predact, temp_det = cal_mul.cal_iou(orgfile, sampled_file)
            iou_list.append(iou)
            print("\n for config = %s the accuracy is = %f" %(config, iou))
        
        zipped = list(zip(config_list, iou_list)) 
        res = sorted(zipped, key = operator.itemgetter(1))
        res1 = list(zip(*res))
        
        out = [item for item in res1[0]] 
        out.reverse()
        out1 = [item for item in res1[1]] 
        out1.reverse()
        index = config_list.index(out[0])
        print("\n SP : frame id = %d, total_config = %d and best config name = %s and best possible accuracy = %f" %(i, len(frame_config_list), out[0], out1[0]))
        best_config_list.append(out[0])
        best_acc_list.append(out1[0])    
                             
    print("\n ======= \n best possible accuracy list \n =========== \n")
    print(best_acc_list)
    print("\n =========== \n per frame best config list \n ============== \n")
    print(best_config_list)
    
if __name__ == "__main__":
	main()
