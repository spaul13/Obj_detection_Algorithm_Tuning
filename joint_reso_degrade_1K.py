import os
checked_fol_list = [3, 4, 6, 10, 13, 17, 19, 23, 25, 28, 31, 33]
length_list = [215, 143, 349, 229, 419, 143, 299, 390, 263, 199, 184, 300]
fol = "H:\\drone_video_cp_" 
new_fol = "H:\\1K\\drone_video_cp_"
for i in range(len(checked_fol_list)):
	src_fol = fol + str(checked_fol_list[i]) + "\\"
	dst_fol = new_fol + str(checked_fol_list[i]) + "\\"
	os.system("mkdir " + dst_fol)
	for j in range(1, length_list[i]+1):
		temp_file = "pic_" + str(j) + "_org.png"
		cmd_str = "ffmpeg -i " + src_fol + temp_file + " -vf scale=1024:-1 " + dst_fol + temp_file
		os.system(cmd_str)
		
		