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
#MAX_NUM = 250 #250 #100
#MAX_NUM = 264
#MAX_NUM = 216
#MAX_NUM = 144
jpeg_enabled = True
org_res = 3840

#all configurations until bw = 5
bw_list = [1,2,5,10,20,30,40,60]#,80]
bw = 5
#configurations allowable between specified and previous here 1-2 Mbps
#bw_list = [2,5,10,20,30,40,60]
#bw = 2 # to reduce number of configs

prefix = "bw_" #bw_tiny_, bw_tiny_416


new_list = []
#config = []
fs = 30
approx_level = 2

period = 1 #avaerage out performance of a configuration over period-many frames


def main():
    print(len(sys.argv))
    config_list = []
    if (len(sys.argv) < 3):
        print("Usage: python program_name directory max_image_number")
    else:
        temp_fol = sys.argv[1]
        MAX_NUM = int(sys.argv[2])
        #just for logging used a different file
        dir = int(sys.argv[3])
        wfile = open("best_config\\best_config_" + str(dir)+ ".txt", "w+")
        print(str(len(sys.argv)) + "," + str(temp_fol))
    for i in range(len(bw_list)):
        if(bw_list[i]>bw):
            break
        else:
            file_list = os.listdir(temp_fol + "\\" +prefix +str(bw_list[i]) +"\\")
            print(len(file_list))
            count = len(file_list)
            temp_count = 0
            for element in file_list:
                if(temp_count<count):
                    temp_element = element[:-4].split("_")
                    x=""
                    for j in range(len(temp_element)-1):
                        if j>0:
                            x+="_"+temp_element[j]
                        else:
                            x+=temp_element[j]
                    temp = temp_fol + "\\" +prefix +str(bw_list[i]) +"\\" + x
                    if x not in config_list:
                        config_list.append(x)
                    if temp not in new_list:
                        new_list.append(temp)   
                    
                    
                     
                
    print(new_list)
    print(len(new_list))
    print("\n the number of configs = %d" %len(config_list))
    ind_list = []
    temp_ind = -1
    count_similar = 0
    diff = []
    trigger = 0
    interval = []
    last_i = -1
    count_good = 0
    acc_diff = []
    best_config_list, best_acc_list = [], []
    for i in range(1,MAX_NUM+2-period):
        iou_list = []
        iou2 = []
        print("\n ======= \n Frame %d \n =========== \n" %i)
        config_list = []
        acc_list = []
        
        for j in range(len(new_list)):
            iou_list1 = []
            for k in range(period):
                sampled_file = new_list[j] + "_" +str(i+k) + ".txt"
                orgfile = temp_fol + "\\bbox_org\\" +  "log_bbox_org_" +  str(i+k) +".txt"
                #iou_array[i][j], mismatch_org[i][j], predorg[i][j], predact[i][j], temp_det = cal_mul.cal_iou(orgfile, sampled_file)
                iou, mismatch_org, predorg, predact, temp_det = cal_mul.cal_iou(orgfile, sampled_file)
                iou_list1.append(iou)
                config = new_list[j].split("\\")[-1]
                config_list.append(config)
                #print(config + "," + str(j))
                acc_list.append(iou)
            iou_list.append(statistics.mean(iou_list1))
        
        temp_config_list = config_list
        zipped = zip(config_list, acc_list) 
        zipped = list(zipped)
        res = sorted(zipped, key = operator.itemgetter(1))
        res1 = list(zip(*res))
        
        out = [item for item in res1[0]] 
        out.reverse()
        out1 = [item for item in res1[1]] 
        out1.reverse()
        index = temp_config_list.index(out[0])
        #top_k_index = out.index(list_map_label[i-201])
        #print("\n predict config is in top %d and config name = %s and accuracy = %f" %(top_k_index, out[top_k_index], out1[top_k_index]))
        print("\n best config id %d and config name = %s and accuracy = %f" %(index, out[0], out1[0]))
        wfile.write("%s\n"%out[0])
        best_config_list.append(out[0])
        best_acc_list.append(out1[0])
    print("\n ======= \n best possible accuracy list \n =========== \n")
    print(best_acc_list)
    print("\n =========== \n all configurations list \n ============== \n")
    print(config_list)
    print("\n =========== \n per frame best config list \n ============== \n")
    print(best_config_list)
		
if __name__ == "__main__":
	main()
