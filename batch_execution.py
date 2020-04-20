import os, statistics
import calculate_iou_2obj as cal_mul
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

fol_list = dir
file_count = total_frame_number
"""
fol_list = [23, 24, 25, 26, 9, 10, 11, 12, 13, 15, 16, 17]
file_count = [390, 360, 263, 261, 441, 230, 300, 420, 420, 144, 240, 144]
fol_list = [35,36,37,38]#[33, 34]#[29, 30, 31, 32]#[27, 28]#[9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21]
file_count = [384,145,180,210]#[300, 325]#[288, 121, 216, 240]#[420, 331]#[441, 230, 300, 420, 420, 144, 240, 144, 300, 330, 360]
"""
"""
for i in range(17,len(fol_list)):
	temp_str = "python ground_truth_expr.py H:\\drone_video_cp_"+str(fol_list[i]) + " bbox_drone_"+str(fol_list[i])+"\\ " + str(file_count[i])
	os.system(temp_str)
"""
"""
fol_list = [27, 28]#[9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21]
file_count = [420, 331]#[441, 230, 300, 420, 420, 144, 240, 144, 300, 330, 360]
for i in range(len(fol_list)):
    temp_str = "python get_groundtruth_bestconfig.py bbox_drone_"+str(fol_list[i]) + " " + str(file_count[i]) + " > best_config_list_" + str(fol_list[i]) + ".txt" 
    print(temp_str)
    os.system(temp_str)
"""
#in order to get the accuracy for mobile execution using yolo-tiny-v3
#"""
for i in range(17,len(fol_list)):
    acc_list, mismatch_list, predorg_list, predact_list, corrdet_list = [], [], [], [], []
    for j in range(1, file_count[i]+1):
        org_file = "bbox_drone_" + str(fol_list[i]) + "\\bbox_org\\log_bbox_org_" + str(j) + ".txt"
        test_file = "bbox_drone_" + str(fol_list[i]) + "\\bbox_tiny\\log_bbox_tiny_" + str(j) + ".txt"
        iou, mismatch_org, predorg, predact, corr_det = cal_mul.cal_iou(org_file, test_file)
        acc_list.append(iou)
        mismatch_list.append(mismatch_org)
        predorg_list.append(predorg)
        predact_list.append(predact)
        corrdet_list.append(corr_det)
    print(acc_list)
    print("\n for Video_%d : the mean accuracy = %f and accuracy stdev = %f"%(fol_list[i], statistics.mean(acc_list), statistics.stdev(acc_list)))
    print("\n for Video_%d : the mean no of predictions in org = %f, and mean # of preds in actual tiny = %f"%(fol_list[i], statistics.mean(predorg_list), statistics.mean(predact_list)))
    print("\n for Video_%d : the mean no of correct predictions = %f, and mean # of mismatched predictions = %f"%(fol_list[i], statistics.mean(corrdet_list), statistics.mean(mismatch_list)))
    
#"""