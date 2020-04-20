import os,sys
import locality_meas as lm

#all_files = os.listdir("android_yolo_dataset/")
#print all_files
"""
dir_path = "ffmpeg -i android_yolo_dataset/" #UCF_dataset/"
file_str = "android_yolo_0319_1.txt"
fp = open(file_str, "w")
loopNum = 0
temp_dir_name = "0319_objdetection_expr/temp_yolo_0319_1"#"temp_dir_UCF_spatial"
pic_str = " " + temp_dir_name +"/pic_%3d.png"
"""
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



#for files in all_files:
for j in range(len(dir)):
    spatial_meas_list, temporal_meas_list, temporal_image, temporal_new_image, I_frame_size = [], [], [], [], []
    for i in range(1,2):#,dir[j]+1):
        parent_dir = "H:\\drone_video_cp_" + str(dir[j])
        print(i)
        [temporal_1, temporal_2, temporal_3, frame_size, spatial_index] = lm.cal_temporal_pairwise(i, parent_dir)
        temporal_image.append(temporal_1)
        temporal_meas_list.append(temporal_2)
        temporal_new_image.append(temporal_3)
        spatial_meas_list.append(spatial_index)
        I_frame_size.append(frame_size)
        #if loopNum > 1 :
        #    sys.exit()
        #below lines only commented for this (used for decoding)
        #os.system("mkdir " + temp_dir_name)
        #os.system(dir_path + files + pic_str)
        """
        [spatial_meas, spatial_list] = lm.cal_spatial(temp_dir_name)
        spatial_meas_list.append(spatial_meas)

        temporal_meas = lm.cal_temporal(temp_dir_name)
        temporal_meas_list.append(temporal_meas)

        #print("video name = %s"%files)
        #fp.write(files)
        print [spatial_list]
        fp.write("--> %f\n"%(spatial_meas))
        fp.write("--> %f\n"%(temporal_meas))
        """
        #os.system("rm -r " + temp_dir_name)#---commented only for this
        #loopNum = loopNum +1 
    print("\n ======= Temporal Locality as Image ========= \n")
    print(temporal_image)
    print("\n ======= Temporal Locality as Video ========= \n")
    print(temporal_meas_list)
    print("\n ======= New temporal Locality (Image) ========= \n")
    print(temporal_new_image)
    print("\n ======= Spatial Locality ========= \n")
    print(spatial_meas_list)
    print("\n ======= I frame sizes (kB) ========= \n")
    print(I_frame_size)

#print temporal_meas_list
#print [sum(temporal_meas_list)/len(temporal_meas_list)]
#print spatial_meas_list
#print [sum(spatial_meas_list)/len(spatial_meas_list)]






