import sys, statistics
import numpy as np
FILE_DIR = "single_image_bbox/"
if len(sys.argv) is not 4:
	print("Usage: program unbuntu_result_file phone_result_file frequency")
else:
    infile1 = sys.argv[1]
    infile2 = sys.argv[2]
    freq = sys.argv[3]

gt = []
bbox = []
count_1, count_2 = 0, 0
with open(infile1, "r") as f:
	for line in f:
		line = line.split()
		# print(line)
		try:
			top = eval(line[0])
			left = eval(line[1])
			bottom = eval(line[2])
			right = eval(line[3])
			gt.append((top, left, bottom, right))
		except:
			gt.append(())

pre = []
with open(infile2, "r") as f:
	for line in f:
		line = line.split()
		top = eval(line[0])
		left = eval(line[1])
		bottom = eval(line[2])
		right = eval(line[3])
		pre.append((top, left, bottom, right))

# print(gt)
for i, g in enumerate(gt):
    if not g:
        continue
    #trying to opportunistically change the frequency depending on previous IOU value
    #"""
    if(freq ==1): 
        count_1+=1
    else:
        count_2+=1
    #"""
    if(i%int(freq)==0):
        p = pre[i]
    #randomly trying to have switch between 1 & 2
    #freq = np.random.choice([1,2], p=[0.3, 0.7])
    if(True):
        top = max(g[0], p[0])
        left = max(g[1], p[1])
        bottom = min(g[2], p[2])
        right = min(g[3], p[3])
        interarea = max(0, bottom - top + 1) * max(0, right - left + 1)
        #Union Area
        b1_area = (g[2] - g[0] +1)*(g[3] - g[1] +1)
        b2_area = (p[2] - p[0] +1)*(p[3] - p[1] +1)
        # print(interarea)
        iou = interarea / (b1_area + b2_area - interarea)
        print(iou)
        bbox.append(iou)
        #trying to opportunistically change the frequency depending on previous IOU value
        """
        if(iou<0.8):
            freq = 1
        else:
            freq = 2
        """
            

print(bbox)
print(sum(bbox)/len(bbox))
print(statistics.stdev(bbox))
#removing noise
bbox_1 = [i for i in bbox if i > 0.8]
print(statistics.mean(bbox_1))
print(statistics.stdev(bbox_1))

print(str(count_1) + "," + str(count_2))
    
 
	