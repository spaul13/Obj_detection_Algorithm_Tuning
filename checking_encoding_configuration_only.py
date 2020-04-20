import os
"""
#os.system("cd bbox_drone_21")
entries = os.listdir("bbox_drone_21\\bw_5\\")
print(entries)
encode_list = []
for i in entries:
    temp = i.split("_")
    if(temp[0]=='encode'):
        if int(temp[1]) not in encode_list:
            encode_list.append(int(temp[1]))
print(encode_list)            
"""
"""
checked_fol_list = [2,3,10,11,17,22,24,25,27,29,31,35,38] 
#length_list = [249, 263, 419, 440, 299, 419, 143, 239, 329, 359, 360, 360, 261, 420, 288, 121, 170, 325]
#accuracy of model on test dataset
checked_fol_list.extend([4,5,9,15,21,23,26,28,30,32,36,37])
#length_list = [215, 143, 349, 229, 419, 143, 299, 390, 263, 199, 184, 300, 384, 145, 180, 210]
checked_fol_list.sort()
print(checked_fol_list)
parent_fol = "bbox_drone_"
bw_list = [1, 2, 5]
for i in range(len(checked_fol_list)):
    fol = parent_fol + str(checked_fol_list[i])
    print("\n =========== \n %s \n =========== \n"%fol)
    encode_list = []
    for j in range(len(bw_list)):
        bw_fol = fol + "\\bw_" + str(bw_list[j]) + "\\"
        entries = os.listdir(bw_fol)
        #print(entries)
        for k in entries:
            temp = k.split("_")
            if(temp[0]=='encode'):
                if int(temp[1]) not in encode_list:
                    encode_list.append(int(temp[1]))
    print(encode_list) 
"""
index_list = [9, 21] #[15, 22, 25]
file_number = [440, 359]#[143, 360, 263]
list1, list2, list3 = [], [], []
encode_list =['encode_28', 'encode_36']
for i in range(len(index_list)):
        for j in range(1, file_number[i]+1):
            list1.append(index_list[i])
            list2.append(j)
            list3.append(encode_list[i])

print(list1)
print()
print(list2)
print()
print(list3)
print(len(list1))


    
		
	
		
	
	