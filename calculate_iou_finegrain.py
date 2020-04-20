import sys
import numpy as np
import statistics
FILE_DIR = "bbox_log/"
if len(sys.argv) is not 4:
	print("Usage: program unbuntu_result_file phone_result_file MAX_NO")
else:
    infile1 = sys.argv[1]
    infile2 = sys.argv[2]
    max_file = int(sys.argv[3])+1

#gt = [[0.0]*5]*max_file
#pre = [[0.0]*5]*max_file
gt = np.zeros((max_file, 5))
pre = np.zeros((max_file,5))
#print(gt[0][:])
mismatch = 0
total, total_org, total_both = 0, 0, 0
bbox_accuracy = []
for i in range(max_file):
    for j in range(5):
        gt[i][j] = -1
        pre[i][j] = -1

with open(infile1, "r") as f:
    for line in f:
        line = line.split()
		# print(line)
        try:
            top, left, bottom, right, class_index, pic_index = eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3]), eval(line[4]), eval(line[5])
            gt[int(pic_index)][0], gt[int(pic_index)][1], gt[int(pic_index)][2], gt[int(pic_index)][3], gt[int(pic_index)][4] = top, left, bottom, right, class_index
            #print(gt[0][:])
        except:
            print("exception\n")
			#gt.append(())

#pre = []
with open(infile2, "r") as f:
    for line in f:
        line = line.split()
        top, left, bottom, right, class_index, pic_index = eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3]), eval(line[4]), eval(line[5])
        #print(pic_index)
        pre[int(pic_index)][0], pre[int(pic_index)][1], pre[int(pic_index)][2], pre[int(pic_index)][3], pre[int(pic_index)][4] = top, left, bottom, right, class_index
        #print("%f, %f, %f, %f, %d" %(pre[0][0], pre[0][1], pre[0][2], pre[0][3], pre[0][4]))
		#pre.append((top, left, bottom, right))

#print(gt[100][0:4])
#for i, g in enumerate(gt):
for i in range(1,max_file):
    #if not g:
    #    continue
    #p = pre[i]
    if((gt[i][0]>=0.0)):
        total_org+=1
    if((pre[i][0]>=0.0)):
        total+=1
    if((gt[i][0]>=0.0) and (pre[i][0]>=0.0)):
        #print(i)
        total_both+=1
        if(gt[i][4] != pre[i][4]):
            mismatch+=1
            print(i)
            print(str(gt[i][4])+","+str(pre[i][4]))
            bbox_accuracy.append(0)
        else:
            top = max(gt[i][0], pre[i][0])
            left = max(gt[i][1], pre[i][1])
            bottom = min(gt[i][2], pre[i][2])
            right = min(gt[i][3], pre[i][3])
            #print(right)
            interarea = max(0, bottom - top + 1) * max(0, right - left + 1)
            #Union Area
            b1_area = (gt[i][2] - gt[i][0] +1)*(gt[i][3] - gt[i][1] +1)
            b2_area = (pre[i][2] - pre[i][0] +1)*(pre[i][3] - pre[i][1] +1)
            #print(interarea)
            #print(interarea / (b1_area + b2_area - interarea))
            bbox_accuracy.append(interarea / (b1_area + b2_area - interarea))

print("\n average bounding box accuracy = %f" %statistics.mean(bbox_accuracy))
print("\n std deviation = %f" %statistics.stdev(bbox_accuracy))
print("\n total prediction in orginal = %d, total prediction in final = %d and final falsely predicted = %d, final overpredicted = %d" %(total_org, total, mismatch, total - total_both))      
print(mismatch)
print(total)
print(total_org)
#print(bbox_accuracy)
        