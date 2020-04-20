import os
#recent version
total_frame_number = [249, 263, 419, 440, 299, 419, 143, 239, 329, 359, 360, 360, 261, 420, 288, 121, 170, 325]
total_frame_number.extend([215, 143, 349, 229, 419, 143, 299, 390, 263, 199, 184, 300])
dir  = [2,3,7,9,11,12,15,16,20,21,22, 24,26,27,29,30,32,34]
dir.extend([4, 5, 6, 10, 13, 17, 19, 23, 25, 28, 31, 33])
dir_sort = sorted(dir)
total_sort = []
print(dir)
print(dir_sort)
print(len(dir),len(total_frame_number))
for i in range(len(dir_sort)):
    total_sort.append(total_frame_number[dir.index(dir_sort[i])])

print(len(dir_sort), len(total_frame_number))

pre_cmd = "python get_groundtruth_bestconfig.py bbox_drone_"
dir_sort = [35, 36, 37, 38]
total_sort = [384,145,180,210]

dir_sort = [25]
total_sort = [263]

for i in range(len(dir_sort)):
	cmd_str = pre_cmd + str(dir_sort[i]) + " " + str(total_sort[i]) + " " + str(dir_sort[i])
	print(cmd_str)
	os.system(cmd_str)
	
