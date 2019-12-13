import os, sys
from os import listdir
from os.path import isfile, join
import statistics
import numpy as np
import calculate_iou_2obj as cal_mul
MAX_NUM = 401 #401 #401 #for jockey #201 for shark #later we can make it as argument or find the maximum number
#Downsample_array = [352, 480, 608, 704] #four different downsample size empericially found (through sampling and testing)
Downsample_array = [384, 544, 672, 768] #four different downsample size empericially found (through sampling and testing) #for jockey
#Encoding_qp = [37, 30, 28, 26] --> test1
#Encoding_qp = [18, 20, 22, 24] --> test2
#Encoding_qp = [22, 24, 28, 37] # used finally for 20, 40, 60, 80
Encoding_qp = [22, 25, 28, 36] # used finally for 20, 40, 60, 80 #for jockey
Jpeg_res = [3840, 1920, 1440]
org_res = 3840 #4K images captured
full_acc_log = "bbox_log\\bbox_4K_original.txt"
outfol = []
freq_array = [2, 3, 4, 6, 8]
iou_array = np.zeros((MAX_NUM, len(freq_array)))
mismatch_org = np.zeros((MAX_NUM, len(freq_array)))
predorg = np.zeros((MAX_NUM, len(freq_array)))
predact = np.zeros((MAX_NUM, len(freq_array)))


#the output feed to network will be downsize-upsize-CNN model
def Downsize(infol):
    #final output file initialization
    #for k in range(len(Downsample_array)):
    #    iou[k] = []
    #    fd[k] = open("final_op_" + str(Downsample_array[k]) + ".txt"
    iou_352, iou_480, iou_608, iou_704 = [], [], [], [] 
    for j in range(len(Downsample_array)):
        outfol.append(infol + "\\" + str(Downsample_array[j]))
    
    for i in range(1, MAX_NUM+1):
        infile = infol + "\pic_" + str(i) + "_org.png"
        for j in range(len(Downsample_array)):
            outfile = infile[:-4] + "_" + str(Downsample_array[j]) + ".png"
            #Downscaling (at client side)
            cmd_str = "ffmpeg -i " + infile +" -vf scale=" + str(Downsample_array[j]) +":-1 " + outfile
            os.system(cmd_str)
            #Upscaling (Server side)
            outfile_1 = outfol[j] + "\pic_" + str(i) + "_" + str(Downsample_array[j]) + "_up.png" #outfile[:-4] + "_up.png"
            cmd_str = "ffmpeg -i " + outfile +" -vf scale=" + str(org_res) +":-1 " + outfile_1
            os.system(cmd_str)
            #deletion
            cmd_str = "rm " + outfile
            os.system(cmd_str)
            """
            #run the cnn model
            bbox_txt = "log_bbox.txt"
            cmd_str = "python detect.py --images " + outfile_1 + " --det det --file_name " + bbox_txt[:-4]
            os.system(cmd_str)
            #run the calculate_iou script to get the bounding box accuracy
            cmd_str = "python calculate_iou.py " + full_acc_log + " " + bbox_txt + " > log.txt"
            with open("log.txt") as fp:
                for line in fp:
                    iou_value = float(line.split(" ")[0])
                    print(iou_value)
                    if(j==0):
                        iou_352.append(iou_value)
                    elif(j==1):
                        iou_480.append(iou_value)
                    elif(j==2):
                        iou_608.append(iou_value)
                    else:
                        iou_704.append(iou_value)   
    """
        


#encode and decode version with different CRF and QP
def Encode(infol):
    
    for j in range(len(Encoding_qp)):
        outfol.append(infol + "\\" + str(Encoding_qp[j]))
    """
    for i in range(1, MAX_NUM+1):
        infile = infol + "\pic_" + str(i) + "_org.png"
        for j in range(len(Encoding_qp)):
            outfile = outfol[j] + "\pic" + "_" + str(i) + "_" + str(Encoding_qp[j]) + ".mp4"
            #encoding (at client side) 
            #ffmpeg -i pic_1_org.png -c:v libx264 -qp 27.5  output_1.mp4
            cmd_str = "ffmpeg -i " + infile +" -c:v libx264 -qp " + str(Encoding_qp[j]) +"  " + outfile
            print(cmd_str)
            os.system(cmd_str)
    """
    #do the detections now
    for j in range(len(Encoding_qp)):
        for i in range(1, MAX_NUM+1):
            infile = infol + "\\" + str(Downsample_array[j]) + "\pic_" + str(i) + "_" +  str(Encoding_qp[j]) + ".mp4"
            outfile = "log_" + str(Encoding_qp[j]) + "_" + str(i)
            #encoding (at client side) 
            #ffmpeg -i pic_1_org.png -c:v libx264 -qp 27.5  output_1.mp4
            cmd_str = "python video_demo.py --video " + infile +" --reso 1024 --file_name " + outfile
            print(cmd_str)
            os.system(cmd_str)        
    

def mul_exp(infol):
    """
    for j in range(len(Downsample_array)):
        #running the cnn model
        fol = infol + "\\" + str(Downsample_array[j])
        onlyfiles = [f for f in listdir(fol) if isfile(join(fol, f))]
        txtfile = "log_4K_" + str(Downsample_array[j]) + ".txt"
        cmd_str = "python detect.py --images " + fol + " --file_name " + txtfile [:-4]
        os.system(cmd_str)
        #run the calculate_iou.py to get the bounding box accuracy
        ioufile = "iou_4K_" + str(Downsample_array[j]) + ".txt"
        print(len(onlyfiles))
        cmd_str = "python calculate_iou_finegrain.py " + full_acc_log + " bbox_log\\" + txtfile  + " " + str(len(onlyfiles)) + " > " + ioufile
        print(cmd_str)
        os.system(cmd_str)
    """  
    for j in range(len(Downsample_array)):
        ioufile = "iou_4K_" + str(Downsample_array[j]) + ".txt"
        print("\n ============= IOU (%d) ============ \n" %(Downsample_array[j]))
        iou = []
        with open(ioufile) as fp:
            for line in fp:
                temp = line.split(' ')[0]
                #print(temp)
                iou.append(float(temp))
        #print(iou)
        print("\n average bounding box accuracy = %f" %statistics.mean(iou))
        print("\n std deviation = %f" %statistics.stdev(iou))
        

def Jpeg_conversion(infol):
    
    for j in range(len(Jpeg_res)):
        outfol.append(infol + "\\" + str(Jpeg_res[j]))
    
    for i in range(1, MAX_NUM+1):
        infile = infol + "\pic_" + str(i) + "_org.png"
        for j in range(len(Jpeg_res)):
            outfile = infile[:-4] + "_" + str(Jpeg_res[j]) + ".jpg"
            #Downscaling (at client side)
            cmd_str = "ffmpeg -i " + infile +" -vf scale=" + str(Jpeg_res[j]) +":-1 " + outfile
            os.system(cmd_str)
            #Upscaling (Server side)
            outfile_1 = outfol[j] + "\pic_" + str(i) + "_" + str(Jpeg_res[j]) + "_up.jpg" #outfile[:-4] + "_up.png"
            cmd_str = "ffmpeg -i " + outfile +" -vf scale=" + str(org_res) +":-1 " + outfile_1
            os.system(cmd_str)
            #deletion
            cmd_str = "rm " + outfile
            os.system(cmd_str)
    """
    for j in range(len(Jpeg_res)):
        #running the cnn model
        fol = infol + "\\" + str(Jpeg_res[j])
        onlyfiles = [f for f in listdir(fol) if isfile(join(fol, f))]
        txtfile = "log_4K_JPEG_" + str(Jpeg_res[j]) + ".txt"
        cmd_str = "python detect.py --images " + fol + " --file_name " + txtfile [:-4]
        os.system(cmd_str)
    
    
        #run the calculate_iou.py to get the bounding box accuracy
        ioufile = "iou_4K_JPEG_" + str(Jpeg_res[j]) + ".txt"
        print(len(onlyfiles))
        cmd_str = "python calculate_iou_finegrain.py " + full_acc_log + " bbox_log\\" + txtfile  + " " + str(len(onlyfiles)) + " > " + ioufile
        print(cmd_str)
        os.system(cmd_str)
    """
#python detect.py --images H:\decoded_jockey\544\pic_1_544_up.png  --det det --file_name bbox_jockey
def new_expr(infol):
    
    """
    for j in range(len(Downsample_array)):
        outfol.append(infol + "\\" + str(Downsample_array[j]))
    for i in range(1, MAX_NUM+1):
        for j in range(len(Downsample_array)):
            outfile = outfol[j] + "\pic_" + str(i) + "_" + str(Downsample_array[j]) + "_up.png"
            bbox_txt = "log_bbox_" + str(Downsample_array[j]) + "_" + str(i) +".txt"
            cmd_str = "python detect.py --images " + outfile + " --reso 1024 --file_name " + bbox_txt[:-4]
            os.system(cmd_str)
    
    
    #for raw frames
    for i in range(1, MAX_NUM+1):
        outfile = infol + "\\raw_frames\pic_" + str(i) + "_org.png"
        bbox_txt = "log_bbox_org_" +  str(i) +".txt"
        cmd_str = "python detect.py --images " + outfile + " --reso 1024 --file_name " + bbox_txt[:-4]
        os.system(cmd_str)
    
    """
    #for JPEG
    """
    for j in range(len(Jpeg_res)):
        outfol.append(infol + "\\" + str(Jpeg_res[j]))
    for i in range(1, MAX_NUM+1):
        for j in range(len(Jpeg_res)):
            outfile = outfol[j] + "\pic_" + str(i) + "_" + str(Jpeg_res[j]) + "_up.jpg"
            bbox_txt = "log_bbox_jpeg_" + str(Jpeg_res[j]) + "_" + str(i) +".txt"
            cmd_str = "python detect.py --images " + outfile + " --reso 1024 --file_name " + bbox_txt[:-4]
            os.system(cmd_str)
    
    """
    #Downsizing
    
    count_bad = np.zeros(4)
    correct_det = np.zeros(4)
    for j in range(len(Jpeg_res)):
    #for j in range(len(Downsample_array)):
        #count_bad=0
        #correct_det = 0
        for i in range(MAX_NUM):
            #print("\n ===== %d ======= \n"%(i+1))
            orgfile = infol + "\log_bbox_org_" +  str(i+1) +".txt"
            #sampled_file = infol + "\log_bbox_" + str(Downsample_array[j]) + "_" + str(i+1) +".txt"
            #sampled_file = infol + "\log_" + str(Encoding_qp[j]) + "_" + str(i+1) +".txt"
            sampled_file = infol + "\log_bbox_jpeg_" + str(Jpeg_res[j]) + "_" + str(i+1) +".txt"
            iou_array[i][j], mismatch_org[i][j], predorg[i][j], predact[i][j], temp_det = cal_mul.cal_iou(orgfile, sampled_file)  
            correct_det[j]+=temp_det
            if(iou_array[i][j] < 0.5): # for recall 0.4 seems to be the best
                count_bad[j]+=1
            if(iou_array[i][j] == 0):
                if(predorg[i][j]>0):
                    iou_array[i][j] = 0.0
                #else:
                #    iou_array[i][j] = 0.9
                #print("\n no detection %d, %d %d"%(predorg[i][j], predact[i][j], mismatch_org[i][j]))
        #print("\n %d. current bad count = %d, and correct detection = %d" %(Downsample_array[j], count_bad[j], correct_det[j]))
        #print("\n %d. current bad count = %d, and correct detection = %d" %(Encoding_qp[j], count_bad[j], correct_det[j]))
        print("\n %d. current bad count = %d, and correct detection = %d" %(Jpeg_res[j], count_bad[j], correct_det[j]))
               
    
    #Encoding
    """
    for i in range(MAX_NUM):
        orgfile = infol + "\log_bbox_org_" +  str(i+1) +".txt"
        print("\n ===== %d ======= \n"%(i+1))
        for j in range(len(Encoding_qp)):
            sampled_file = infol + "\log_" + str(Encoding_qp[j]) + "_" + str(i+1) +".txt"
            iou_array[i][j], mismatch_org[i][j], predorg[i][j], predact[i][j] = cal_mul.cal_iou(orgfile, sampled_file)        
            if(iou_array[i][j] == 0):
                if(predorg[i][j]>0):
                    iou_array[i][j] = 0.0
                else:
                    iou_array[i][j] = 0.9
    """
    #frequency
    """
    count_bad = np.zeros(len(freq_array))
    correct_det = np.zeros(len(freq_array))    
    for j in range(len(freq_array)):
        for i in range(MAX_NUM):
            orgfile = infol + "\log_bbox_org_" +  str(i+1) +".txt"
            #print("\n ===== %d ======= \n"%(i+1))
            if((i+1)%freq_array[j] == 1):
                sampled_file = infol + "\log_bbox_org_" +  str(i+1) +".txt"
            iou_array[i][j], mismatch_org[i][j], predorg[i][j], predact[i][j], temp_det = cal_mul.cal_iou(orgfile, sampled_file)  
            correct_det[j]+=temp_det
            if(iou_array[i][j] < 0.5): # for recall 0.4 seems to be the best
                count_bad[j]+=1
            if(iou_array[i][j] == 0):
                if(predorg[i][j]>0):
                    iou_array[i][j] = 0.0
                #else:
                #    iou_array[i][j] = 0.9
                #print("\n no detection %d, %d %d"%(predorg[i][j], predact[i][j], mismatch_org[i][j]))
        #print("\n %d. current bad count = %d, and correct detection = %d" %(Downsample_array[j], count_bad[j], correct_det[j]))
        #print("\n %d. current bad count = %d, and correct detection = %d" %(Encoding_qp[j], count_bad[j], correct_det[j]))
        print("\n %d. current bad count = %d, and correct detection = %d" %(freq_array[j], count_bad[j], correct_det[j]))
    """
    
    temp_mean = np.mean(iou_array, axis = 0)
    temp_stdev = np.std(iou_array, axis = 0)
    temp_num = np.count_nonzero(iou_array, axis = 0)
    temp_mismatch = np.sum(mismatch_org, axis = 0)
    temp_predorg = np.sum(predorg, axis = 0)
    temp_predact = np.sum(predact, axis = 0)
    print(temp_mean)
    #for j in range(len(freq_array)):
    #for j in range(len(Downsample_array)):
    for j in range(len(Jpeg_res)):
        TP = correct_det[j] - count_bad[j]
        FN = count_bad[j]
        FP = temp_predact[j] - correct_det[j]
        Recall = float(TP)/(TP+FN)
        Precision = float(TP)/(TP+FP)
        F1 = 2.0*(Precision*Recall)/(Precision + Recall)
        mAP1 = float(TP)/temp_predorg[j]
        mAP2 = float(TP)/temp_predact[j]
        
        print("\n ================ \n statistics for downsample %d \n ================= \n"%(Jpeg_res[j]))
        print("\n TP: %d, FN: %d, FP: %d \n" %(TP, FN, FP))
        print("\n Recall: %f, Precision: %f, F1 : %f \n" %(Recall, Precision, F1))
        print("\n mAP1: %f, mAP2: %f \n" %(mAP1, mAP2))
        print("\n mean = %f"%(temp_mean[j]))#/temp_num[j]))
        print("\n std deviation %f"%(temp_stdev[j]))
        print("\n total prediction in orginal = %d, total prediction in actual = %d, total mismatch prediction  = %f " %(temp_predorg[j], temp_predact[j], temp_mismatch[j]))
       
      
    


def main():
    print(len(sys.argv))
    if len(sys.argv) is not 2:
        print("Usage: program image_directory")
    else:
        infol = sys.argv[1]
        print(str(len(sys.argv)) + "," + str(infol))
        #Downsize(infol)
        #mul_exp(infol)
        #Encode(infol)
        #Jpeg_conversion(infol)
        new_expr(infol)
        #Encode(infol)







if __name__ == "__main__":
	main()


