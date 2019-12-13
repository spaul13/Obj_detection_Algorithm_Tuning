import os
#for spatial
encode_spatial = "x264 --crf 23 --tune fastdecode --fps 60 --keyint 1 --min-keyint 1 --no-scenecut --input-res 4096*2048 --log-level debug --bframes 0  -o temp_video/"
encode_temporal = "x264 --crf 23 --tune fastdecode --fps 60 --keyint 10000000 --min-keyint 10000000 --no-scenecut --input-res 4096*2048 --log-level debug --bframes 0  -o temp_video.mp4 0319_objdetection_expr/temp_yolo_0319_5/pic_%3d.png"
delim = ".png"

encode_temporal_pair = "x264 --crf 17 --tune fastdecode --fps 60 --keyint 2 --min-keyint 2 --no-scenecut --input-res 4096*2048 --log-level debug --bframes 0  -o temp_video.mp4 "
encode_spatial_new = "x264 --crf 17 --tune fastdecode --fps 60 --keyint 2 --min-keyint 2 --no-scenecut --input-res 4096*2048 --log-level debug --bframes 0  -o temp_video_1.mp4 temporal_4K/pic_1.png 2>log_0917.txt"
#--crf 23 --qp 23

def size(folder_str):
    p = os.listdir(folder_str)
    total_size = 0.0
    for i in p:
        stat_info = os.stat(folder_str + i)
        total_size += stat_info.st_size
    return total_size


folder_str = "Furion_Encoder/"
def cal_spatial(dir_name):
    q= os.listdir(dir_name + "/")
    os.system("mkdir temp_video")
    count = 0
    each_entry = []
    for j in q:
        if delim in j:
            count += 1
            temp_str = j.partition(delim)[0]
            print [count]
            spatial_str1 = encode_spatial + temp_str + ".mp4  " + dir_name +"/" + j
            os.system(spatial_str1)
            modified_size = (os.stat("temp_video/"+temp_str+".mp4")).st_size
            unmodified_size = (os.stat( dir_name +"/" + j)).st_size
            each_entry.append(modified_size/unmodified_size)

    #calculate the size of the previous directory only 
    unmodified_size = size(dir_name + "/")
    #calculate the size of the directory with encoded videos only 
    modified_size = size("temp_video/")
    spatial_locality = modified_size/unmodified_size
    #print [spatial_locality]
    #return [spatial_locality, each_entry]
    os.system("rm -r temp_video/")

def cal_temporal(dir_name):
    q= os.listdir(dir_name + "/")
    os.system(encode_temporal)
    #calculate the size of the previous directory only 
    unmodified_size = size(dir_name + "/")
    #calculate the size of the directory with encoded videos only
    stat_info = os.stat("temp_video.mp4")
    modified_size = stat_info.st_size
    temporal_locality = modified_size/unmodified_size
    print [temporal_locality]
    return temporal_locality

def find_size():
    size = 0
    with open("log_temp.txt") as fp:
	for line in fp:
	    print line
            p = line.split("size=")
            size = float((p[len(p)-1].split("bytes"))[0])
            print("\n the current size = " + str(size))
    return size

def cal_temporal_pairwise(i):
    temporal_dir = "temporal_4K"
    if(True):
        os.system("mkdir " +temporal_dir)
        os.system("pwd")
        os.system("scp ../cnn_videos/pic_" + str(i) + ".png temporal_4K/pic_0.png")
        os.system("scp ../cnn_videos/pic_" + str(i+1) + ".png temporal_4K/pic_1.png")
        #os.system("x264 --crf 23 --tune fastdecode --fps 60 --keyint 1 --min-keyint 1 --no-scenecut --input-res 4096*2048 --log-level debug --bframes 0  -o temp_video.mp4 temporal_4K/pic_0.png")
        os.system(encode_spatial_new)
        os.system("grep 'Slice:I' log_0917.txt > log_temp.txt")
        I_frame_size_only = find_size()
        os.system(encode_temporal_pair + temporal_dir + "/pic_%d.png 2>log_0917.txt")
        os.system("grep 'Slice:P' log_0917.txt > log_temp.txt")
        P_frame_size = find_size()
        os.system("grep 'Slice:I' log_0917.txt > log_temp.txt")
        I_frame_size = find_size()
        stat_info = os.stat("temporal_4K/pic_0.png")
        filesize_1 = stat_info.st_size
        stat_info = os.stat("temporal_4K/pic_1.png")
        filesize_2 = stat_info.st_size
        temporal_1 = P_frame_size / filesize_2
        temporal_2 = (I_frame_size + P_frame_size)/(filesize_1 + filesize_2)
        spatial = I_frame_size_only/filesize_1
        temporal_3 = P_frame_size / I_frame_size_only
        os.system("rm -r " +temporal_dir)
        #return [temporal_1, temporal_2, spatial]
        return [temporal_1, temporal_2, temporal_3, I_frame_size_only/1024, spatial]




        





