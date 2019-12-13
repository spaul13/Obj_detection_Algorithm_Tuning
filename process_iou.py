import sys
#from bbox import bbox_iou_sp
import numpy as np
if len(sys.argv) is not 3:
	print("Usage: program unbuntu_result_file phone_result_file")
else:
	infile1 = sys.argv[1]
	infile2 = sys.argv[2]

#box_1 = []

#this was initially written in order to utilize the already available function in bbox.py but that requires torch.tensor processing 
#need to do torch.tensor(numpy_array).cuda() if cuda is available

with open(infile1, "r") as f:
    top_list = []
    left_list = []
    bottom_list = []
    right_list = []
    count = 0
    for line in f:
        line = line.split()
        print(line)
        top_list.append(eval(line[0])) #x1
        left_list.append(eval(line[1])) #y1
        bottom_list.append(eval(line[2])) #x2
        right_list.append(eval(line[3])) #y2
        count+=1
    box_1 = np.zeros((count, 4))
    
    box_1[:,0] = np.asarray(top_list)
    box_1[:,1] = np.asarray(left_list)
    box_1[:,2] = np.asarray(bottom_list)
    box_1[:,3] = np.asarray(right_list)
    
        
#print(box_1[:,0])


with open(infile2, "r") as f:
    top_list = []
    left_list = []
    bottom_list = []
    right_list = []
    count = 0
    for line in f:
        line = line.split()
        print(line)
        top_list.append(eval(line[0])) #x1
        left_list.append(eval(line[1])) #y1
        bottom_list.append(eval(line[2])) #x2
        right_list.append(eval(line[3])) #y2
        count+=1
    box_2 = np.zeros((count, 4))
    
    box_2[:,0] = np.asarray(top_list)
    box_2[:,1] = np.asarray(left_list)
    box_2[:,2] = np.asarray(bottom_list)
    box_2[:,3] = np.asarray(right_list)
    

print(bbox_iou_sp(box_1, box_2))

		