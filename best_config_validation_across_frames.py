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
MAX_NUM = 250 #100
jpeg_enabled = True
org_res = 3840
#bw_list = [1,2,5,10,15,20,30,40,60,80]
bw_list = [1,2,5,10,20,30,40,60]#,80]

prefix = "bw_" #bw_tiny_, bw_tiny_416
bw = 5
new_list = []
#config = []
fs = 30
approx_level = 2
#period = 1 #for general scenario
period = 1 #avaerage out performance of a configuration over period-many frames
#f = open("labels_yolo_911_1.txt", "w+")
f_class = open("best_class_911_check.txt", "w+")
#for 26K iterations
#['tennisracket', 'pottedplant', 'banana', 'banana', 'banana', 'banana', 'donut', 'donut', 'banana', 'banana', 'donut', 'apple', 'apple', 'sandwich', 'banana', 'banana', 'banana', 'donut', 'banana', 'donut', 'donut', 'banana', 'tennisracket', 'keyboard', 'banana', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'tennisracket', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'person', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard']
#list_map_label = ['reso_1440_encode_40', 'reso_640_jpeg_encode_32', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_640_encode_28', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_28', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_640_encode_28', 'reso_480_encode_20', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_480_encode_20', 'reso_1440_encode_40', 'reso_960_jpeg_encode_36', 'reso_480_encode_20', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_1440_encode_40', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_160_encode_20', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36', 'reso_960_jpeg_encode_36']
#for 10K iterations
#list_map_label = ['reso_480_encode_24', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_24', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_24', 'reso_640_encode_28', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_640_encode_28', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_160_encode_20', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_20', 'reso_480_encode_20']
#for 5K iterations
#list_map_label = ['reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_640_encode_28', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_20', 'reso_480_encode_24', 'reso_480_encode_20', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_640_encode_28', 'reso_480_encode_28', 'reso_1440_encode_40', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24', 'reso_480_encode_24']
#for 2.6K iterations (convnet with three layers for 7.6K iterations 200 training samples)
#list_map_label = ['reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20', 'reso_480_encode_20']
#convnet with three layers 10K iteration 100 training samples
list_map_label = ['reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_1440_encode_40', 'reso_160_encode_20', 'reso_160_encode_20', 'reso_160_encode_20', 'reso_160_encode_20', 'reso_160_encode_20', 'reso_160_encode_20', 'reso_160_encode_20', 'reso_160_encode_20', 'reso_160_encode_20', 'reso_160_encode_20']
def main():
    print(len(sys.argv))
    if (len(sys.argv) < 2):
        print("Usage: program image_directory")
    else:
        temp_fol = sys.argv[1]
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
                    if temp not in new_list:
                        new_list.append(temp)
                    """
                    if x not in config:
                        config.append(x)
                    """
                    
                    
                     
                
    print(new_list)
    print(len(new_list))
    """ #writing the labels
    for q in range(len(new_list)):
        temp_config = new_list[q].split("//")
        f.write("%d, %s \n" %(q,temp_config[-1]))
    """
    ind_list = []
    temp_ind = -1
    count_similar = 0
    diff = []
    trigger = 0
    interval = []
    last_i = -1
    count_good = 0
    acc_diff = []
    for i in range(201,MAX_NUM+1-period):
    #for i in range(1,MAX_NUM+1-period):
        #item_list = []
        iou_list = []
        iou2 = []
        print("\n ======= \n Frame %d \n =========== \n" %i)
        config_list = []
        acc_list = []
        
        for j in range(len(new_list)):
            iou_list1 = []
            for k in range(period):
                sampled_file = new_list[j] + "_" +str(i+k) + ".txt"
                #orgfile = "bbox_drone\\bbox_org\\" +  "log_bbox_org_" +  str(i+k) +".txt"
                orgfile = "bbox_drone_2\\bbox_org\\" +  "log_bbox_org_" +  str(i+k) +".txt"
                #iou_array[i][j], mismatch_org[i][j], predorg[i][j], predact[i][j], temp_det = cal_mul.cal_iou(orgfile, sampled_file)
                iou, mismatch_org, predorg, predact, temp_det = cal_mul.cal_iou(orgfile, sampled_file)
                iou_list1.append(iou)
                config = new_list[j].split("\\")[-1]
                config_list.append(config)
                acc_list.append(iou)
                #print(str(new_list[j]) + "," +config + "," + str(iou))
                #item_list.append(item)
            iou_list.append(statistics.mean(iou_list1))
        
        zipped = zip(config_list, acc_list) 
        zipped = list(zipped)
        res = sorted(zipped, key = operator.itemgetter(1))
        res1 = list(zip(*res))
        
        out = [item for item in res1[0]] 
        out.reverse()
        out1 = [item for item in res1[1]] 
        out1.reverse()
        top_k_index = out.index(list_map_label[i-201])
        print("\n predict config is in top %d and config name = %s and accuracy = %f" %(top_k_index, out[top_k_index], out1[top_k_index]))
        print("\n best config id 0 and config name = %s and accuracy = %f" %(out[0], out1[0]))
        acc_diff.append(out1[0] - out1[top_k_index])
        
        if(top_k_index < 10):
            count_good +=1
        
  

        """    
        actual = []
        
        for j in range(int((len(iou2))/period)):
            temp_actual = []
            for k in range(period):
                temp_actual.append(iou2[j+(k*period)])
            actual.append(max(temp_actual))
        """
    
        """
        ind = iou_list.index(max(iou_list))
        print(ind)
        if(temp_ind == ind):
            count_similar+=1
        temp_ind = ind
        ind_list.append(ind)
        print(new_list[ind])
        print(iou_list[ind])
        temp_config = new_list[ind].split("\\")
        #f_class.write("%d, %d, %s, %f\n" %(i, ind, temp_config[-1], iou_list[ind]))
        """
        """
        #if(((i-1)%approx_level == 0) or (trigger==1)):
        if(trigger==1) or (i==1):
            approx_ind = temp_ind
            trigger = 0
            if(last_i >=0):
                interval.append(i-last_i)
            last_i = i
        print(new_list[approx_ind])
        print(iou_list[approx_ind])
        diff.append(iou_list[ind] - iou_list[approx_ind])
        if(diff[len(diff)-1] > 0.05):
            trigger = 1
        """
    #print("\n number of consecutive frames having similar configurations " + str(count_similar))
    print("\n good predictions are = %d" %count_good)
    print("\n accuracy difference = %f" %(statistics.mean(acc_diff)))
    
    
    
    
    #print("\n mean difference in accuracy = %f, std dev = %f" %(statistics.mean(diff), statistics.stdev(diff)))
    #print("\n mean difference in interval of execution = %f, std dev = %f" %(statistics.mean(interval), statistics.stdev(interval)))
    """
    colors = ['r', 'g', 'c', 'm', 'y', 'b', 'k']
    name = "Dark2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    color1 = cmap.colors
    marker_list = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    for ele in color1:
        colors.append(ele)
    
    
    for i in range(MAX_NUM): 
        if(i%20 == 0):
            col = colors[int(i/20)]
            #video_prog = (i+1)*50
        plt.scatter(i+1, ind_list[i], c = col, marker = 'o',  s=100) 
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel('Frame Indices', fontsize = fs)
    #plt.ylabel('IOU Accuracy', fontsize = fs)
    plt.ylabel('Best Configuration Index', fontsize = fs)
    plt.show()
	"""	
		
		
		
		
if __name__ == "__main__":
	main()
