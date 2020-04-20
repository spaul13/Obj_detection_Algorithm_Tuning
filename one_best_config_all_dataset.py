from os import path
import calculate_iou_2obj as cal_mul
import statistics
import numpy as np
reso_list = [2048, 1440, 960, 768, 640, 480, 320, 160]
qp_list = np.linspace(12,40,8)


total_frame_number = [249, 263, 419, 440, 299, 419, 143, 239, 329, 359, 360, 360, 261, 420, 288, 121, 170, 325]
total_frame_number.extend([215, 143, 349, 229, 419, 143, 299, 390, 263, 199, 184, 300])
dir  = [2,3,7,9,11,12,15,16,20,21,22, 24,26,27,29,30,32,34]
dir.extend([4, 5, 6, 10, 13, 17, 19, 23, 25, 28, 31, 33])
dir_sort = sorted(dir)
total_sort = []
#print(dir)
#print(dir_sort)
#print(len(dir),len(total_frame_number))
for i in range(len(dir_sort)):
    total_sort.append(total_frame_number[dir.index(dir_sort[i])])
total_frame_number = total_sort
dir = dir_sort

#extended partition
dir.extend([35, 36, 37, 38])
total_frame_number.extend([384,145,180,210])

bw_list = [1,2,5,10,20,30,40,60,80]
bw = 5
bw_index = bw_list.index(bw)
print(bw_index)
bbox_str = "bbox_drone_"


def read_file(filename):
    f5 = open(filename, "r+")
    new_list = []
    for line in f5:
        temp = " ".join(line.split())
        new_list.append(temp)
        #print(temp)
        #print(len(temp))
    return new_list

def intersection(list1, list2):
    list3 = [value for value in list1 if value in list2]
    return list3


def cal_diff(intersect_config_list):
    iou_config_diff, iou_config_variation = [], []
    total_config_list = intersect_config_list
    #"""
    for i in range(len(total_config_list)):
        print("\n =========== %d. current configuration = %s ======== \n"%(i, total_config_list[i]))
        temp_iou, prev_iou, new_iou = [], [], []
        for l in range(len(dir)):
            dir_index = dir[l]
            best_config_list = read_file("best_config\\best_config_" + str(dir_index) +".txt")
            for j in range(total_frame_number[l]): #for j in range(total_frame_number[index]-1):
                sampled_file = ""
                for l in range(bw_index+1):
                    if(path.exists(bbox_str+ str(dir_index) + "\\bw_" + str(bw_list[l])+"\\" + best_config_list[j] + "_" + str(j+1) + ".txt")):
                        best_file = bbox_str+ str(dir_index) + "\\bw_" + str(bw_list[l])+"\\" + best_config_list[j] + "_" + str(j+1) + ".txt"
                        #print("\n best config file = " + best_file)
                    if(path.exists(bbox_str+ str(dir_index) + "\\bw_" +str(bw_list[l])+"\\" + total_config_list[i] + "_" + str(j+1) + ".txt")):
                        sampled_file = bbox_str+ str(dir_index) + "\\bw_" +str(bw_list[l])+"\\" + total_config_list[i] + "_" + str(j+1) + ".txt"
                        #print("\n sampled file = " + sampled_file)
                orgfile = bbox_str+ str(dir_index) + "\\bbox_org\\log_bbox_org_" + str(j+1) + ".txt"
                iou1, mismatch_org1, predorg1, predact1, temp_det1 = cal_mul.cal_iou(orgfile, best_file)
                iou2, mismatch_org2, predorg2, predact2, temp_det2 = cal_mul.cal_iou(orgfile, sampled_file)
                temp_iou.append(iou1 - iou2)
                new_iou.append(iou1)
                prev_iou.append(iou2)
                #print("\n current iou difference = " + str(iou1 - iou2))
        iou_config_diff.append(statistics.mean(temp_iou))
        iou_config_variation.append(statistics.stdev(temp_iou))
    
    best_single_configuration = total_config_list[iou_config_diff.index(min(iou_config_diff))]
    print("\n the single best video configuration %s and overall mean accuracy degradation = %d" %(best_single_configuration,min(iou_config_diff)))
    
    #Now need to log impact of that single best configuration on each dataset
    total_config_list = [best_single_configuration]
    for i in range(len(total_config_list)):
        #print("\n =========== current configuration = %s ======== \n"%i)
        temp_iou, prev_iou, new_iou = [], [], []
        for l in range(len(dir)):
            print("\n =========== Video index = %d ======== \n"%dir[l])
            dir_index = dir[l]
            best_config_list = read_file("best_config\\best_config_" + str(dir_index) +".txt")
            for j in range(total_frame_number[l]): #for j in range(total_frame_number[index]-1):
                sampled_file = ""
                for l in range(bw_index+1):
                    if(path.exists(bbox_str+ str(dir_index) + "\\bw_" + str(bw_list[l])+"\\" + best_config_list[j] + "_" + str(j+1) + ".txt")):
                        best_file = bbox_str+ str(dir_index) + "\\bw_" + str(bw_list[l])+"\\" + best_config_list[j] + "_" + str(j+1) + ".txt"
                        #print("\n best config file = " + best_file)
                    if(path.exists(bbox_str+ str(dir_index) + "\\bw_" +str(bw_list[l])+"\\" + total_config_list[i] + "_" + str(j+1) + ".txt")):
                        sampled_file = bbox_str+ str(dir_index) + "\\bw_" +str(bw_list[l])+"\\" + total_config_list[i] + "_" + str(j+1) + ".txt"
                        #print("\n sampled file = " + sampled_file)
                orgfile = bbox_str+ str(dir_index) + "\\bbox_org\\log_bbox_org_" + str(j+1) + ".txt"
                iou1, mismatch_org1, predorg1, predact1, temp_det1 = cal_mul.cal_iou(orgfile, best_file)
                iou2, mismatch_org2, predorg2, predact2, temp_det2 = cal_mul.cal_iou(orgfile, sampled_file)
                temp_iou.append(iou1 - iou2)
                new_iou.append(iou1)
                prev_iou.append(iou2)
                #print("\n current iou difference = " + str(iou1 - iou2))
            #iou_config_diff.append(statistics.mean(temp_iou))
            #iou_config_variation.append(statistics.stdev(temp_iou))
            print("\n best single configuration = %s, mean accuracy = %f, accuracy deviation = %f" %(total_config_list[i], statistics.mean(temp_iou),statistics.stdev(temp_iou)))
	
def main():
    intersect_list = read_file("total_config\\config_" + str(dir[0]) +".txt")
    for i in range(1, len(dir)):
        print("\n ==== %d ===== \n" %i)
        temp_list = read_file("total_config\\config_" + str(dir[i]) +".txt")
        intersect_list = intersection(intersect_list, temp_list)
        print(len(intersect_list))
    print(intersect_list)
    print(len(intersect_list))
    cal_diff(intersect_list)
		



if __name__ == "__main__":
	main()