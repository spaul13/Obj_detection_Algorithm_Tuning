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
spatial_meas_list = []
temporal_meas_list = []
temporal_image = []
temporal_new_image = []
I_frame_size = []

#for files in all_files:
for i in range(1,401):
    [temporal_1, temporal_2, temporal_3, frame_size, spatial_index] = lm.cal_temporal_pairwise(i)
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
print temporal_image
print("\n ======= Temporal Locality as Video ========= \n")
print temporal_meas_list
print("\n ======= New temporal Locality (Image) ========= \n")
print temporal_new_image
print("\n ======= Spatial Locality ========= \n")
print spatial_meas_list
print("\n ======= I frame sizes (kB) ========= \n")
print I_frame_size

#print temporal_meas_list
#print [sum(temporal_meas_list)/len(temporal_meas_list)]
#print spatial_meas_list
#print [sum(spatial_meas_list)/len(spatial_meas_list)]






