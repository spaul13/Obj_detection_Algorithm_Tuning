import cv2, os

#drone+diving videos
trainset = [2,3,9,11,17,21,24,25,27,29,30,32,35,38]
train_fol_size = [251,264,441,300,144,360,360,263,420,288,121,170,384,210]

trainset = [2,3,11,17,21,24,28,29,30,32,38]
train_fol_size = [251,264,300,144,360,360,199,288,121,170,210]

testset = [4,5,10,15,22,23,26,28,31,36,37]
test_fol_size = [216,144,230,144,360,390,261,199,184,145,180]

#trainset = testset
out_fol = "H:\\new_training_reduced_set\\"
counter = 0
#first check the resolution
"""
for i in range(len(trainset)):
    pic_name = "H:\\drone_video_cp_" + str(trainset[i]) + "\\pic_1_org.png"
    im = cv2.imread(pic_name)
    h, w, c = im.shape
    print("\n Dataset %d: Image resolution = [%d,%d]" %(trainset[i],w,h))
    if((w==3840) and (h==2160)):
        for j in range(1,train_fol_size[i]+1):
            os.system("mkdir " + out_fol + str(counter))
            infile = "H:\\drone_video_cp_" + str(trainset[i]) + "\\pic_" + str(j) + "_org.png"
            outfile = out_fol + str(counter) + "\\pic_" + str(counter) + "_org.png"
            os.system("scp " + infile + " " + outfile)
            counter+=1
    else:
        for j in range(1,train_fol_size[i]+1):
            os.system("mkdir " + out_fol + str(counter))
            infile = "H:\\drone_video_cp_" + str(trainset[i]) + "\\pic_" + str(j) + "_org.png"
            outfile = out_fol + str(counter) + "\\pic_" + str(counter) + "_org.png"
            cmd_str = "ffmpeg -i " + infile +" -vf scale=3840:2160 " + outfile
            os.system(cmd_str)
            counter+=1
"""
trainset = [28]
train_fol_size = [199]

for i in range(len(trainset)):
    fol_name = "H:\\drone_video_cp_" + str(trainset[i]) + "\\pic_"
    log_fol = "bbox_drone_" + str(trainset[i]) + "\\bbox_org\\log_bbox_org_"
    for j in range(1,train_fol_size[i]+1):
        orgfile = fol_name + str(j) + "_org.png"
        cmd_str = "python detect.py --images " + orgfile + " --reso 1024 --file_name " + log_fol + str(j)
        os.system(cmd_str)

"""
def outside(msg):
    print(msg)

class A():
    def __init__(self, i):
        self.msg = i
    def method(self):
        outside(self.msg)

a = A("check")
a.method()
"""
    
        
        


