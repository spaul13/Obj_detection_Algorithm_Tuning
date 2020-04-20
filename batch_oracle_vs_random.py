from os import path
import calculate_iou_2obj as cal_mul
import statistics
import numpy as np
reso_list = [2048, 1440, 960, 768, 640, 480, 320, 160]
qp_list = np.linspace(12,40,8)



#unique_best_config_list = list(np.unique(np.array(best_config_list)))

#recent version
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
dir = [35, 36, 37, 38]
total_frame_number = [384,145,180,210]

bw_list = [1,2,5,10,20,30,40,60,80]
bw = 5
bw_index = bw_list.index(bw)
print(bw_index)
bbox_str = "bbox_drone_"

#index = 0

#find what single QP or resolution is used if encoding or downsizing algorithm used only
def find_sofart_downsize(total_config_list):
    for i in total_config_list:
        for j in range(len(reso_list)):
            temp = "reso_" + str(reso_list[j])
            if(temp in total_config_list):
                #print(temp)
                return temp
                break
        else:
            continue
        break
    return ''
def find_sofart_encode(total_config_list):
    for i in total_config_list:
        for j in range(len(qp_list)):
            temp = "encode_" + str(int(qp_list[j]))
            if(temp in total_config_list):
                #print(temp)
                return temp
                break
        else:
            continue
        break
    
    return ''
    



def cal_diff(total_config_list, best_config_list,total_frame_number, dir_index):
    iou_config_diff, iou_config_variation = [], []
    print("\n ========== \n drone_video_cp_%d \n ========== \n" %dir_index)
    #"""
    for i in range(len(total_config_list)):
        #print("\n =========== current configuration = %s ======== \n"%i)
    #for i in unique_best_config_list:
        temp_iou, prev_iou, new_iou = [], [], []
        for j in range(total_frame_number): #for j in range(total_frame_number[index]-1):
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
    
    """
    print("\n ====== state-of-the-art iou ====== \n")
    print(prev_iou)
    print("\n ======== in our method oracle iou ======== \n")
    print(new_iou)
    print("\n average iou difference between oracle and state-of-the art ===")
    print(iou_config)
    #print("\n current config list ")
    #print(total_config_list)
    #print(len(iou_config))
    """
    for i in range(len(total_config_list)):
        print("\n current configuration %s, avg accuracy = %f, accuracy deviation = %f "%(total_config_list[i], iou_config_diff[i], iou_config_variation[i]))
    #comparison with the state-of-the-art mechanisms
    if(find_sofart_downsize(total_config_list)!=''):
        sofart_downsize = find_sofart_downsize(total_config_list)
        accuracy_diff = iou_config_diff[total_config_list.index(sofart_downsize)]
        print("\n state of the art downsizing = %s, accuracy difference with oracle = %f" %(sofart_downsize,accuracy_diff))
    else:
        print("\n no state-of-the-art downsizing possible with this bw constraint")
    if(find_sofart_encode(total_config_list)!=''):
        sofart_encode = find_sofart_encode(total_config_list)
        accuracy_diff = iou_config_diff[total_config_list.index(sofart_encode)]
        print("\n state of the art encoding = %s, accuracy difference with oracle = %f" %(sofart_encode,accuracy_diff))
    else:
        print("\n no state-of-the-art encoding possible with this bw constraint")
        
    
    worst_single_config = total_config_list[iou_config_diff.index(max(iou_config_diff))]
    best_single_config = total_config_list[iou_config_diff.index(min(iou_config_diff))]
    print("\n max iou diff = %f, and worst single config = %s" %(max(iou_config_diff), worst_single_config))
    print("\n min iou diff = %f, and best single config = %s" %(min(iou_config_diff), best_single_config))
    print("\n mean = %f, stddev = %f" %(statistics.mean(iou_config_diff),statistics.stdev(iou_config_diff)))
	#"""

def read_file(filename):
    f5 = open(filename, "r+")
    new_list = []
    for line in f5:
        temp = " ".join(line.split())
        new_list.append(temp)
        #print(temp)
        #print(len(temp))
    return new_list

def main():
    for i in range(len(dir)):
        total_config_list = read_file("total_config\\config_" + str(dir[i]) +".txt")
        best_config_list = read_file("best_config\\best_config_" + str(dir[i]) +".txt")
        print(len(total_config_list), len(best_config_list))
        cal_diff(total_config_list, best_config_list,total_frame_number[i], dir[i])


if __name__ == "__main__":
	main()

				
					
	
				
		
		
	