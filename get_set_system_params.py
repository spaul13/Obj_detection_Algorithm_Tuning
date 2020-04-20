import psutil, time, os
dir = "H:\\drone_video_cp_11\\pic_"
parent = "H:\\drone_video_cp_"
jpeg_qp = 18
encoding_qp = 25 
def A(count):
	for i in range(count):
		print()
trainset = [2,3,9,11,17,21,24,25,27,29,30,32,35,38]        
thread_count = [0, 1, 2, 3, 4, 5, 6, 7, 8]
number_tested = 50
def main():
    p=psutil.Process()
    p.nice(psutil.HIGH_PRIORITY_CLASS)#Normal priority --> 8, high priority --> 13 #increasing priority helps?
    for k in range(len(trainset)):
        dir = parent + str(trainset[k])
        print("\n ================== \n %s \n ============== \n" %dir)
        dir = dir + "\\pic_"
        thread_time, thread_time_hwaccel, thread_time_dxva2 = [], [], []
        for j in range(len(thread_count)):
            duration, duration_hwaccel, duration_dxva2 = 0.0, 0.0, 0.0
            for i in range(number_tested):
                #start_time = time.time()
                infile = dir + str(i+2) + "_org.png"
                cmd_str = "ffmpeg -threads " + str(thread_count[j]) + " -i " + infile +" -vf scale=960:-1 -q:v "+ str(jpeg_qp) +" -c:v libx264 -qp " + str(encoding_qp) + " -loglevel quiet -y temp_1.mp4"
                start_time = time.time()
                os.system(cmd_str)
                duration += (time.time() - start_time)
                #available hwaccel qsv , dxva2 (not cuda)
                #qsv provides better performance (than normal and dxva2)
                #thread count depends on content
                cmd_str = "ffmpeg -hwaccel qsv -threads " + str(thread_count[j]) + " -i " + infile +" -vf scale=960:-1 -q:v "+ str(jpeg_qp) +" -c:v libx264 -qp " + str(encoding_qp) + " -loglevel quiet -y temp_1.mp4"
                #os.system(cmd_str)
                start_time = time.time()
                os.system(cmd_str)
                duration_hwaccel += (time.time() - start_time)
                cmd_str = "ffmpeg -hwaccel qsv -threads " + str(thread_count[j]) + " -i " + infile +" -vf scale=960:-1 -q:v "+ str(jpeg_qp) +" -c:v libx264 -qp " + str(encoding_qp) + " -loglevel quiet -y temp_1.mp4"
                start_time = time.time()
                os.system(cmd_str)
                duration_dxva2 += (time.time() - start_time)
            thread_time.append(duration/number_tested)
            thread_time_hwaccel.append(duration_hwaccel/number_tested)
            thread_time_dxva2.append(duration_dxva2/number_tested)
            #print("\n total time taken :%f" %(duration/10))
        for j in range(len(thread_count)):
            print("No HW accel: # of threads = %d, avg time taken = %f ms \n"%(thread_count[j],thread_time[j]*1000))
            print("QSV: # of threads = %d, avg time taken = %f ms \n"%(thread_count[j],thread_time_hwaccel[j]*1000))
            print("DXVA2: # of threads = %d, avg time taken = %f ms \n \n"%(thread_count[j],thread_time_dxva2[j]*1000))
    """
    print(p.nice())
    for i in range(10000):
        print()
    print(p.cpu_percent(interval=None))
    """
    #A(1000000)

if __name__== "__main__":
  main()