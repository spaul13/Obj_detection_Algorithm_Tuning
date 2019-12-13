import numpy as np
import matplotlib.pyplot as plt
import statistics, os, sys
import calculate_iou_2obj as cal_mul
from matplotlib.cm import get_cmap
#reso_list = [3200, 2560, 2048, 1440, 960, 768, 640, 480, 320, 160]
reso_list = [2048, 1440, 960, 768, 640, 480, 320, 160]
#qp_list = np.linspace(0,40,11)
qp_list = np.linspace(12,40,8)
#qp_list = [36, 40]
#qp_list = np.linspace(24,40,5)
MAX_NUM = 420 #350 #144 #264 #144 #216 #264 #250
jpeg_enabled = True
org_res = 3840

sz_list = []
mech_list = []

fs = 30
count_acc = []






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
    
    size_kb = sum/(MAX_NUM*1024)
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
    # traverse in the list1 
    """
    print(list1)
    print(lb)
    print(ub)
    """
    for x in range(len(list1)): 
        # condition check 
        if list1[x]>= lb and list1[x]< ub: 
            #print(mech_list[x])
            #start of detection
            fol_in = fol+mech_list[x]
            temp_1 = fol_in.split("\\")[-1]
            ret_list.append(bw_str + "\\" + temp_1)
            
            
            
            print(fol_in)
            temp = fol_in.split("\\")[-2]
            temp_1 = fol_in.split("\\")[-1]
            #print(temp)
            #print(temp_1)
            
            if(c>=0):
                if((temp == "jpeg_encode") or (temp =="encode")):
                    for i in range(1, MAX_NUM+1):
                        infile = fol_in + "\\pic_" + str(i) + "_org.mp4"
                        outfile = bw_str + "\\" + temp_1 + "_" + str(i)
                        cmd_str = "python video_demo.py --video " + infile +" --reso 1024 --file_name " + outfile
                        #cmd_str = "python detect.py --images " + infile + " --file_name " + outfile #for tiny-yolo only
                        print(cmd_str)
                        os.system(cmd_str)
           
                elif("jpeg" in fol_in):
                    print("inside elif")
                    temp_1 = fol_in.split("\\")[-1]
                    for i in range(1,MAX_NUM+1):
                        infile = fol_in + "\\pic_" + str(i) + "_org.jpg"
                        outfile = bw_str + "\\" + temp_1 + "_" + str(i)
                        cmd_str = "python detect.py --images " + infile + " --reso 1024 --file_name " + outfile
                        #cmd_str = "python detect.py --images " + infile + " --file_name " + outfile #for tiny-yolo only
                        print(cmd_str)
                        os.system(cmd_str)
                else:
                    print("inside else")
                    temp_1 = fol_in.split("\\")[-1]
                    for i in range(1,MAX_NUM+1):
                        infile = fol_in + "\\pic_" + str(i) + "_org.png"
                        outfile = bw_str + "\\" + temp_1 + "_" + str(i)
                        cmd_str = "python detect.py --images " + infile + " --reso 1024 --file_name " + outfile
                        #cmd_str = "python detect.py --images " + infile + " --file_name " + outfile #for tiny-yolo only
                        print(cmd_str)
                        os.system(cmd_str)
            
                    
            #end of detection    
                    
             
               
                    
            c+= 1
    
    print("\n total configurations are %d"%c)
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
        #for raw original frames
        """
        for i in range(1, MAX_NUM+1):
            infile = temp_fol + "\\pic_" + str(i) + "_org.png"
            outfile = "bbox_org\\log_bbox_org_" + str(i)
            cmd_str = "python detect.py --images " + infile + " --reso 1024 --file_name " + outfile
            print(cmd_str)
            os.system(cmd_str)
        """
        
        #this part is to generated multiple compressed images
        """
        #Downsizing
        for i in range(len(reso_list)):
            out_fol =  str(sys.argv[1]) + "\\downsize\\reso_" + str(reso_list[i])
            os.system("mkdir " + out_fol)
            Downsize(temp_fol, out_fol, reso_list[i])
            
        #JPEG conversion
        out_fol = str(sys.argv[1]) + "\\jpeg"    
        os.system("mkdir " + out_fol)
        convert_jpeg(temp_fol, out_fol, 0)
        
        #Downsize_jpeg
        for i in range(len(reso_list)):
            out_fol =  str(sys.argv[1]) + "\\downsize_jpeg\\reso_" + str(reso_list[i]) + "_jpeg"
            in_fol = str(sys.argv[1]) + "\\downsize\\reso_" + str(reso_list[i])
            os.system("mkdir " + out_fol)
            convert_jpeg(in_fol, out_fol, 1)
        
        #PNG(full_scale+encode)
        for i in range(len(qp_list)):
            out_fol =  str(sys.argv[1]) + "\\encode\\encode_" + str(int(qp_list[i]))
            os.system("mkdir " + out_fol)
            encode(temp_fol, out_fol, qp_list[i], 0)
     
        #JPEG(full_scale+encode)    
        for i in range(len(qp_list)):
            out_fol =  str(sys.argv[1]) + "\\jpeg_encode\\jpeg_encode_" + str(int(qp_list[i]))
            in_fol = str(sys.argv[1]) + "\\jpeg"
            os.system("mkdir " + out_fol)
            encode(in_fol, out_fol, qp_list[i], 1)
        
        
        #Downscale+Encode
        for i in range(len(reso_list)):
            for j in range(len(qp_list)):
                out_fol =  str(sys.argv[1]) + "\\downsize_encode\\" + "reso_" + str(reso_list[i]) + "_encode_" + str(int(qp_list[j]))
                in_fol = str(sys.argv[1]) + "\\downsize\\reso_" + str(reso_list[i])
                os.system("mkdir " + out_fol)
                down_encode(in_fol, out_fol, qp_list[j], 0)
                
        
        #Downscale+JPEG+Encode
        for i in range(len(reso_list)):
            for j in range(len(qp_list)):
                out_fol =  str(sys.argv[1]) + "\\downsize_jpeg_encode\\" + "reso_" + str(reso_list[i]) + "_jpeg_encode_" + str(int(qp_list[j])) 
                in_fol = str(sys.argv[1]) + "\\downsize_jpeg\\reso_" + str(reso_list[i]) + "_jpeg"
                os.system("mkdir " + out_fol)
                down_encode(in_fol, out_fol, qp_list[j], 1)
        
        """
        
        #"""
        #this part in order to check the generated filesize
        for i in range(3):
            if(i==0):
                for j in range(len(reso_list)):
                    temp_file =  str(sys.argv[1]) + "\\downsize\\reso_" + str(reso_list[j])
                    print("\n ========== \n Downsize (PNG) %d  (in KB) \n ============ \n" %reso_list[j])
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
                    print("\n ===========\n Encode (QP) %d (in KB) \n ============ \n" %qp_list[j])
                    mech_list.append("\\encode\\encode_" + str(int(qp_list[j])))
                    sz_list.append(size(temp_file, 3))
        #PNG_Downsize+Encode
        for i in range(len(reso_list)):
            for j in range(len(qp_list)):
                #if ((reso_list[i]<960) and (qp_list[j]> 12)):
                #    continue
                temp_file =  str(sys.argv[1]) + "\\downsize_encode\\" + "reso_" + str(reso_list[i]) + "_encode_" + str(int(qp_list[j]))
                print("\n ===========\n PNG Downsize %d Encode (QP) %d (in KB) \n ============ \n" %(reso_list[i],qp_list[j]))
                mech_list.append("\\downsize_encode\\reso_" + str(reso_list[i]) + "_encode_" + str(int(qp_list[j])))
                sz_list.append(size(temp_file, 2))
        #PNG_Downsize+JPEG+Encode
        for i in range(len(reso_list)):
            for j in range(len(qp_list)):
                #if ((reso_list[i]<2048) and (qp_list[j]> 12)):
                #    continue
                temp_file =  str(sys.argv[1]) + "\\downsize_jpeg_encode\\" + "reso_" + str(reso_list[i]) + "_jpeg_encode_" + str(int(qp_list[j]))
                print("\n ===========\n PNG Downsize %d + JPEG + Encode (QP) %d (in KB) \n ============ \n" %(reso_list[i],qp_list[j]))
                mech_list.append("\\downsize_jpeg_encode\\reso_" + str(reso_list[i]) + "_jpeg_encode_" + str(int(qp_list[j])))
                sz_list.append(size(temp_file, 2))
        #JPEG+Encode
        for j in range(len(qp_list)):
            temp_file =  str(sys.argv[1]) + "\\jpeg_encode\\" + "jpeg_encode_" + str(int(qp_list[j]))
            print("\n ===========\n JPEG + Encode (QP) %d (in KB) \n ============ \n" %(qp_list[j]))
            mech_list.append("\\jpeg_encode\\jpeg_encode_" + str(int(qp_list[j])))
            sz_list.append(size(temp_file,3))
        #"""
        
        #"""
        #used for parallel execution only
        ub = int(sys.argv[2])
        lb = int(sys.argv[3])
        print(str(lb) + "," +str(ub))
        scale = 8.75 #scale in order to convert the bw to the allowable filesize can be transmitted (in 70 ms), rest is detection time for full yolo
        #scale = 10.25 #for tiny yolo, detection time = 18 ms
        #print(len(list2))
        print("\n =================== \n configurations possible under %d Mbps BW \n ======================== \n" %ub)
        bw_str = "bw_" + str(ub)
        os.system("mkdir bbox_drone\\" + bw_str)
        temp = count(sz_list, lb*scale, ub*scale, temp_fol, bw_str)
        print(temp)
        print(len(temp))
        #"""
        """
        bw_list = [1,2,5,10,20,30,40,60]#,80]
        #scale = 8.75
        scale = 10.25
        #this part is used while plotting graphs
        for i in range(len(bw_list)):
            print("\n =================== \n configurations possible under %d Mbps BW \n ======================== \n" %bw_list[i])
            bw_str = "bw_tiny_" + str(bw_list[i])
            #os.system("mkdir bbox_drone\\" + bw_str)
            ub = bw_list[i]
            if(i==0):
                lb = 0
            else:
                lb = bw_list[i-1]
            temp = count(sz_list, lb*scale, ub*scale, temp_fol, bw_str)
            #temp = count(sz_list, (bw_list[i]*scale)/1.2, (bw_list[i]*scale)*1.14, temp_fol, bw_str)
            print(temp)
            
            
            if(len(list2)>=0):
                #list3 = list(set(temp) - set(list2))#[value for value in temp if value not in list2] 
                for ele_list in list2:
                    temp.append(ele_list)
                list3 = temp
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
                    #temp_sp = list3[j].split("\\")[-1]
                    temp_sp = list3[j]
                    print(temp_sp)
                    for i in range(0, MAX_NUM):
                        #sampled_file = "bbox_drone\\" + bw_str + "\\" + temp_sp + "_" + str(i+1) +".txt"
                        sampled_file = "bbox_drone\\" + temp_sp + "_" + str(i+1) +".txt"
                        orgfile = "bbox_drone\\bbox_org\\" +  "log_bbox_org_" +  str(i+1) +".txt"
                        iou_array[i][j], mismatch_org[i][j], predorg[i][j], predact[i][j], temp_det = cal_mul.cal_iou(orgfile, sampled_file)  
                        #print(iou_array[i][j])
                        correct_det[j]+=temp_det
                        #if(iou_array[i][j] < 0.5): # for recall 0.4 seems to be the best
                        #    count_bad[j]+=1
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
                
                #total_accuracy.append(temp_mean)
                max_list3 = max(list3)
                ta = []
                for j in range(len(list3)):
                    TP = correct_det[j] - count_bad[j]
                    FN = count_bad[j]
                    FP = temp_predact[j] - correct_det[j]
                    if(TP+FN > 0):
                        Recall = float(TP)/(TP+FN)
                    else:
                        Recall = 0.0
                    if(TP+FN > 0):
                        Precision = float(TP)/(TP+FP)
                    else:
                        Precision = 0.0
                    if(Precision + Recall > 0):
                        F1 = 2.0*(Precision*Recall)/(Precision + Recall)
                    else:
                        F1 = 0.0
                    if(temp_predorg[j] > 0):
                        mAP1 = float(TP)/temp_predorg[j]
                    else:
                        mAP1 = 0.0
                    if(temp_predact[j] > 0):
                        mAP2 = float(TP)/temp_predact[j]
                    else:
                        mAP2 = 0.0
                    temp_sp = list3[j].split("\\")[-1]
                    if(list3[j] == max_list3):
                        print("\n the best configuration %s, and accuracy =%f" %(temp_sp, float(temp_mean[j])))
                    #in order to get the F1 score
                    #ta.append(F1)
                    ta.append(temp_mismatch[j]/250)
                    print("\n ================ \n %d . statistics for current config %s \n ================= \n"%(j, temp_sp))
                    print("\n TP: %d, FN: %d, FP: %d \n" %(TP, FN, FP))
                    print("\n Recall: %f, Precision: %f, F1 : %f \n" %(Recall, Precision, F1))
                    print("\n mAP1: %f, mAP2: %f \n" %(mAP1, mAP2))
                    print("\n mean = %f"%(temp_mean[j]))#/temp_num[j]))
                    print("\n std deviation %f"%(temp_stdev[j]))
                    print("\n total prediction in orginal = %d, total prediction in actual = %d, total mismatch prediction  = %f " %(temp_predorg[j], temp_predact[j], temp_mismatch[j]))
                    
            total_accuracy.append(np.asarray(ta))
            #list2=temp
            list2 = list3
            #list2 = []
            
    
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
    #plt.ylabel('IOU Accuracy', fontsize = fs)
    #plt.ylabel('F1 Score', fontsize = fs)
    plt.ylabel('Fraction of No/Miss Predictions', fontsize = fs)
    plt.show()
    """          
        
        
                
if __name__ == "__main__":
	main()

				
				
		
				
				
			



