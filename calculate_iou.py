import sys
FILE_DIR = "bbox_log/"
if len(sys.argv) is not 3:
	print("Usage: program unbuntu_result_file phone_result_file")
else:
	infile1 = sys.argv[1]
	infile2 = sys.argv[2]

gt = []
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
    p = pre[i]
    top = max(g[0], p[0])
    left = max(g[1], p[1])
    bottom = min(g[2], p[2])
    right = min(g[3], p[3])
    interarea = max(0, bottom - top + 1) * max(0, right - left + 1)
    #Union Area
    b1_area = (g[2] - g[0] +1)*(g[3] - g[1] +1)
    b2_area = (p[2] - p[0] +1)*(p[3] - p[1] +1)
    # print(interarea)
    print(interarea / (b1_area + b2_area - interarea))
    
 
	