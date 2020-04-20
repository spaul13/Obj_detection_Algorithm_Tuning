import os, glob
bw_list = [1,2,5,10]
dir_index = 2
curr_bw = 5
bw_index = bw_list.index(curr_bw)
dir  = [2,3,7,9,11,12,15,16,20,21,22, 24,26,27,29,30,32,34]
dir.extend([4, 5, 6, 10, 13, 17, 19, 23, 25, 28, 31, 33])
dir_sort = sorted(dir)
dir_sort = [35, 36, 37, 38]
for dir_index in dir_sort:
    dirname = "bbox_drone_" + str(dir_index)
    config_list = []
    wfile = open("total_config\\config_" + str(dir_index)+ ".txt", "w+")
    print("\n ======= \n %s \n ========= \n" %dirname)
    for j in range(bw_index+1):
        bw_file = dirname + "\\bw_" + str(bw_list[j])
        print(bw_file)
        file_list = glob.glob(bw_file + "\\*.txt")
        #print(file_list)
        for i in file_list:
            file_name = i.split("\\")[-1]
            frag = file_name.split("_")
            config = frag[0]
            for k in range(1,len(frag)-1):
                config += "_" + frag[k]
            if config not in config_list:
                config_list.append(config)
                wfile.write("%s\n"%config)
                #print(config)

    print(config_list)
    print(len(config_list))
