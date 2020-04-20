import glob
srcdir = "C:\\yolo\\darknet\\build\\darknet\\x64\\data\\new_training\\"
srcdir = "H:\\new_training_reduced_set\\"

for i in range(2667):
	files = glob.glob(srcdir+str(i)+"\\*.png")
	if len(files)==0:
		print(i)
	