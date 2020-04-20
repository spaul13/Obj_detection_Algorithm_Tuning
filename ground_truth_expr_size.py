import numpy as np
import matplotlib.pyplot as plt
import statistics, os, sys
import calculate_iou_2obj as cal_mul
from matplotlib.cm import get_cmap
import statistics
reso_list = [3200, 2560, 2048, 1440, 960, 768, 640, 480, 320, 160]
qp_list = np.linspace(0,40,11)
MAX_NUM = 100
jpeg_enabled = True
org_res = 3840

sz_list = []
mech_list = []

fs = 30





def Downsize(infol, outfol, reso):
    for i in range(1, MAX_NUM+1):
        infile = infol + "\pic_" + str(i) + "_org.png"
        outfile = outfol + "\pic_" + str(i) + "_temp.png"
        #Downscaling (at client side)
        cmd_str = "ffmpeg -i " + infile +" -vf scale=" + str(int(reso)) +":-1 " + outfile
        os.system(cmd_str)
        #Upscaling (Server side)
        outfile_1 = outfol + "\pic_" + str(i) + "_org.png"
        cmd_str = "ffmpeg -i " + outfile +" -vf scale=" + str(org_res) +":-1 " + outfile_1
        os.system(cmd_str)
        #deletion
        #cmd_str = "rm " + outfile
        #os.system(cmd_str)
		

def convert_jpeg(infol, outfol, order):
    for i in range(1, MAX_NUM+1):
        if(order==1):
            infile = infol + "\pic_" + str(i) + "_temp.png"
        else:
            infile = infol + "\pic_" + str(i) + "_org.png"
        outfile = outfol + "\pic_" + str(i) + "_temp.jpg"
        #jpeg compression on client side
        cmd_str = "ffmpeg -i " + infile + "  " + outfile
        os.system(cmd_str)
        #Upscaling (Server side)
        outfile_1 = outfol + "\pic_" + str(i) +"_org.jpg" 
        cmd_str = "ffmpeg -i " + outfile +" -vf scale=" + str(org_res) +":-1 " + outfile_1
        os.system(cmd_str)

def encode(infol, outfol, qp, jp):
    for i in range(1, MAX_NUM+1):
        outfile = outfol + "\pic_" + str(i) + "_org.mp4"
        if(jp==0):
            infile = infol + "\pic_" + str(i) + "_org.png"
        else:
            infile = infol + "\pic_" + str(i) + "_org.jpg"
        cmd_str = "ffmpeg -i " + infile +" -c:v libx264 -qp " + str(qp) +"  " + outfile
        os.system(cmd_str)
        
def size(infol, type): #type = 0 
    sum = 0
    for i in range(1, MAX_NUM+1):
        if(type==0):
            outfile = infol + "\pic_" + str(i) + "_temp.png"
        elif(type==1):
            outfile = infol + "\pic_" + str(i) + "_temp.jpg"
        elif(type==2):
            outfile = infol + "\pic_" + str(i) + "_temp.mp4"
        else: #this is only encoding or jpeg+encoding
            outfile = infol + "\pic_" + str(i) + "_org.mp4"
        statinfo = os.stat(outfile)
        sum+=statinfo.st_size
    
    size_kb = sum/(100*1024)
    #print(size_kb)
    return size_kb
        
def down_encode(infol, outfol, qp, jp):
    for i in range(1, MAX_NUM+1):
        outfile = outfol + "\pic_" + str(i) + "_temp.mp4"
        if(jp==0):
            infile = infol + "\pic_" + str(i) + "_temp.png"
        else:
            infile = infol + "\pic_" + str(i) + "_temp.jpg"
        #encoding
        cmd_str = "ffmpeg -i " + infile +" -c:v libx264 -qp " + str(qp) +"  " + outfile
        os.system(cmd_str)        
        #decoding
        cmd_str = "ffmpeg -i "  + outfile + " temp.png"
        if(jp==0):
            temp_file = infol + "\\temp.png"
            cmd_str = "ffmpeg -i "  + outfile + "  " + temp_file
            os.system(cmd_str)
            outfile_1 = outfol + "\pic_" + str(i) + "_org.png"
            cmd_str = "ffmpeg -i " + temp_file  + " -vf scale=" + str(org_res) +":-1 " + outfile_1
            os.system(cmd_str)
            os.system("rm " + temp_file)
        else:
            temp_file = infol + "\\temp.jpg"
            cmd_str = "ffmpeg -i "  + outfile + "  " + temp_file
            os.system(cmd_str)
            outfile_1 = outfol + "\pic_" + str(i) + "_org.jpg"
            cmd_str = "ffmpeg -i " + temp_file  + " -vf scale=" + str(org_res) +":-1 " + outfile_1
            os.system(cmd_str)
            os.system("rm " + temp_file)


def perf_detection(fol_in):
    if("encode" in fol_in):
        temp = fol_in.split("\\")[-1]
        print(temp)
       

def count(list1, lb, ub, fol, bw_str):
    #return len(list(x for x in list1 if lb <= x <= ub)) 
    c = 0 
    ret_list = []
    size_sum = []
    # traverse in the list1 
    for x in range(len(list1)): 
        # condition check 
        if list1[x]>= lb and list1[x]<= ub: 
            #print(mech_list[x])
            #start of detection
            fol_in = fol+mech_list[x]
            temp_1 = fol_in.split("\\")[-1]
            ret_list.append(temp_1)
            print("\n current config = %s, tx size = %f" %(temp_1, list1[x]))
            
            size_sum.append(list1[x])
            """
            print(fol_in)
            temp = fol_in.split("\\")[-2]
            #temp_1 = fol_in.split("\\")[-1]
            #print(temp)
            #print(temp_1)
            if((temp == "jpeg_encode") or (temp =="encode")):
                for i in range(1, MAX_NUM+1):
                    infile = fol_in + "\\pic_" + str(i) + "_org.mp4"
                    outfile = bw_str + "\\" + temp_1 + "_" + str(i)
                    cmd_str = "python video_demo.py --video " + infile +" --reso 1024 --file_name " + outfile
                    print(cmd_str)
                    os.system(cmd_str)
           
            elif("jpeg" in fol_in):
                print("inside elif")
                temp_1 = fol_in.split("\\")[-1]
                for i in range(1,MAX_NUM+1):
                    infile = fol_in + "\\pic_" + str(i) + "_org.jpg"
                    outfile = bw_str + "\\" + temp_1 + "_" + str(i)
                    cmd_str = "python detect.py --images " + infile + " --reso 1024 --file_name " + outfile
                    print(cmd_str)
                    os.system(cmd_str)
            else:
                print("inside else")
                temp_1 = fol_in.split("\\")[-1]
                for i in range(1,MAX_NUM+1):
                    infile = fol_in + "\\pic_" + str(i) + "_org.png"
                    outfile = bw_str + "\\" + temp_1 + "_" + str(i)
                    cmd_str = "python detect.py --images " + infile + " --reso 1024 --file_name " + outfile
                    print(cmd_str)
                    os.system(cmd_str)
            
            """        
            #end of detection    
                    
               
               
                    
            c+= 1
    
    print("\n total configurations are %d"%c)
    print("\n maximum size = %f, min size = %f, mean = %f" %(max(size_sum), min(size_sum), statistics.mean(size_sum)))
    return ret_list
    #return c 


def main():
    print(len(sys.argv))
    total_accuracy = []
    if (len(sys.argv) < 2):
        print("Usage: program image_directory")
    else:
        temp_fol = sys.argv[1]
        print(str(len(sys.argv)) + "," + str(temp_fol))
        list2 = []
        #this part in order to check the generated filesize
        for i in range(3):
            if(i==0):
                for j in range(len(reso_list)):
                    temp_file =  str(sys.argv[1]) + "\\downsize\\reso_" + str(reso_list[j])
                    #print("\n ========== \n Downsize (PNG) %d  (in KB) \n ============ \n" %reso_list[j])
                    mech_list.append("\\downsize\\reso_" + str(reso_list[j]))
                    sz_list.append(size(temp_file, 0))
            elif(i==1):
                for j in range(len(reso_list)):
                    temp_file =  str(sys.argv[1]) + "\\downsize_jpeg\\reso_" + str(reso_list[j]) + "_jpeg"
                    #print("\n ========== \n Downsize (JPEG) %d (in KB) \n ============ \n" %reso_list[j])
                    mech_list.append("\\downsize_jpeg\\reso_" + str(reso_list[j]) + "_jpeg")
                    sz_list.append(size(temp_file, 1))
                temp_file = str(sys.argv[1]) + "\\jpeg"
                print("\n ========== \n JPEG (full size = 3840) (in KB) \n ============ \n")
                mech_list.append("\\jpeg")
                sz_list.append(size(temp_file, 1))
            else:
                for j in range(len(qp_list)):
                    temp_file =  str(sys.argv[1]) + "\\encode\\encode_" + str(int(qp_list[j]))
                    #print("\n ===========\n Encode (QP) %d (in KB) \n ============ \n" %qp_list[j])
                    mech_list.append("\\encode\\encode_" + str(int(qp_list[j])))
                    sz_list.append(size(temp_file, 3))
        #PNG_Downsize+Encode
        for i in range(len(reso_list)):
            for j in range(len(qp_list)):
                #if ((reso_list[i]<960) and (qp_list[j]> 12)):
                #    continue
                temp_file =  str(sys.argv[1]) + "\\downsize_encode\\" + "reso_" + str(reso_list[i]) + "_encode_" + str(int(qp_list[j]))
                #print("\n ===========\n PNG Downsize %d Encode (QP) %d (in KB) \n ============ \n" %(reso_list[i],qp_list[j]))
                mech_list.append("\\downsize_encode\\reso_" + str(reso_list[i]) + "_encode_" + str(int(qp_list[j])))
                sz_list.append(size(temp_file, 2))
        #PNG_Downsize+JPEG+Encode
        for i in range(len(reso_list)):
            for j in range(len(qp_list)):
                #if ((reso_list[i]<2048) and (qp_list[j]> 12)):
                #    continue
                temp_file =  str(sys.argv[1]) + "\\downsize_jpeg_encode\\" + "reso_" + str(reso_list[i]) + "_jpeg_encode_" + str(int(qp_list[j]))
                #print("\n ===========\n PNG Downsize %d + JPEG + Encode (QP) %d (in KB) \n ============ \n" %(reso_list[i],qp_list[j]))
                mech_list.append("\\downsize_jpeg_encode\\reso_" + str(reso_list[i]) + "_jpeg_encode_" + str(int(qp_list[j])))
                sz_list.append(size(temp_file, 2))
        #JPEG+Encode
        for j in range(len(qp_list)):
            temp_file =  str(sys.argv[1]) + "\\jpeg_encode\\" + "jpeg_encode_" + str(int(qp_list[j]))
            #print("\n ===========\n JPEG + Encode (QP) %d (in KB) \n ============ \n" %(qp_list[j]))
            mech_list.append("\\jpeg_encode\\jpeg_encode_" + str(int(qp_list[j])))
            sz_list.append(size(temp_file,3))
        
        bw_list = [1,2,5,10,15,20,30,40,60,80]
        scale = 8.75 #scale in order to convert the bw to the allowable filesize can be transmitted (in 70 ms), rest is detection time for full yolo
        #scale = 10.25 #for tiny yolo, detection time = 18 ms
        print(len(list2))
        for i in range(len(bw_list)):
            print("\n =================== \n configurations possible under %d Mbps BW \n ======================== \n" %bw_list[i])
            #count(sz_list, (bw_list[i]*scale)/1.5, (bw_list[i]*scale)*1.14)
            bw_str = "bw_tiny_" + str(bw_list[i])
            #os.system("mkdir bbox_groundtruth\\" + bw_str)
            temp = count(sz_list, (bw_list[i]*scale)/1.5, (bw_list[i]*scale)*1.14, temp_fol, bw_str)
            #temp = count(sz_list, (bw_list[i]*scale)/1.2, (bw_list[i]*scale)*1.14, temp_fol, bw_str)
            """
            if(len(list2)>=0):
                list3 = list(set(temp) - set(list2))#[value for value in temp if value not in list2] 
                #list3 = [value for value in temp if value in list2] #in order to find the overlap
                #print(len(list3))
                #list3 = temp
                #print(list3)
                #list4 = list3.copy()
                #to calculate accuracy (prev: delete overlapping files)
                count_bad = np.zeros(len(list3))
                correct_det = np.zeros(len(list3))
                iou_array = np.zeros((MAX_NUM, len(list3)))
                mismatch_org = np.zeros((MAX_NUM, len(list3)))
                predorg = np.zeros((MAX_NUM, len(list3)))
                predact = np.zeros((MAX_NUM, len(list3)))
                
                for j in range(len(list3)):
                    temp_sp = list3[j].split("\\")[-1]
                    for i in range(0, MAX_NUM):
                        sampled_file = "bbox_groundtruth\\" + bw_str + "\\" + temp_sp + "_" + str(i+1) +".txt"
                        orgfile = "bbox_4K\\" +  "log_bbox_org_" +  str(i+1) +".txt"
                        iou_array[i][j], mismatch_org[i][j], predorg[i][j], predact[i][j], temp_det = cal_mul.cal_iou(orgfile, sampled_file)  
                        #print(iou_array[i][j])
                        correct_det[j]+=temp_det
                        if(iou_array[i][j] < 0.5): # for recall 0.4 seems to be the best
                            count_bad[j]+=1
                        if(iou_array[i][j] == 0):
                            if(predorg[i][j]>0):
                                iou_array[i][j] = 0.0
                
                temp_mean = np.mean(iou_array, axis = 0)
                temp_stdev = np.std(iou_array, axis = 0)
                temp_num = np.count_nonzero(iou_array, axis = 0)
                temp_mismatch = np.sum(mismatch_org, axis = 0)
                temp_predorg = np.sum(predorg, axis = 0)
                temp_predact = np.sum(predact, axis = 0)
                #print(temp_mean)
                
                total_accuracy.append(temp_mean)
                max_list3 = max(list3)
                for j in range(len(list3)):
                    TP = correct_det[j] - count_bad[j]
                    FN = count_bad[j]
                    FP = temp_predact[j] - correct_det[j]
                    Recall = float(TP)/(TP+FN)
                    Precision = float(TP)/(TP+FP)
                    F1 = 2.0*(Precision*Recall)/(Precision + Recall)
                    mAP1 = float(TP)/temp_predorg[j]
                    mAP2 = float(TP)/temp_predact[j]
                    temp_sp = list3[j].split("\\")[-1]
                    if(list3[j] == max_list3):
                        print("\n the best configuration %s, and accuracy =%f" %(temp_sp, float(temp_mean[j])))
                    
                    print("\n ================ \n %d . statistics for current config %s \n ================= \n"%(j, temp_sp))
                    print("\n TP: %d, FN: %d, FP: %d \n" %(TP, FN, FP))
                    print("\n Recall: %f, Precision: %f, F1 : %f \n" %(Recall, Precision, F1))
                    print("\n mAP1: %f, mAP2: %f \n" %(mAP1, mAP2))
                    print("\n mean = %f"%(temp_mean[j]))#/temp_num[j]))
                    print("\n std deviation %f"%(temp_stdev[j]))
                    print("\n total prediction in orginal = %d, total prediction in actual = %d, total mismatch prediction  = %f " %(temp_predorg[j], temp_predact[j], temp_mismatch[j]))
                    
            
            list2 = temp
            
    #https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib/4971431   
    colors = ['r', 'g', 'c', 'm', 'y', 'b', 'k']
    name = "Dark2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    color1 = cmap.colors
    marker_list = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    for ele in color1:
        colors.append(ele)
    new = 0
    for xe, ye in zip(bw_list, total_accuracy):
        plt.scatter([xe] * len(ye), ye, c = colors[new], marker = marker_list[new],  s=100)    
        new+=1
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(bw_list)
    plt.xlabel('Available Bandwidth (Mbps)', fontsize = fs)
    plt.ylabel('IOU Accuracy', fontsize = fs)
    plt.show()
    """            
        
        
                
if __name__ == "__main__":
	main()

				
				
		
				
				
			



