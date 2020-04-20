import os
import glob
file_list = glob.glob("C:\\yolo\\darknet\\build\\darknet\\x64\\data\\obj_0204_prev\\*.png")
org_dir = "C:\\yolo\\darknet\\build\\darknet\\x64\\data\\obj_0204_prev\\"
counter=1
for i in file_list:
    file_name = i.split("\\")[-1]
    os.system("mkdir " + org_dir + "456\\" + str(counter))
    cmd_str = "scp " + org_dir + file_name + " " + org_dir + "456\\" + str(counter) + "\\pic_" + str(counter) + "_org.png"
    print(cmd_str)
    os.system(cmd_str)
    counter+=1
    