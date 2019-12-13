import os

f1 = open("labels_yolo_911_1.txt", "r+")
f2 = open("best_class_911.txt", "r+")
f3 = open("final_labels_911.txt", "w+")
f4 = open("train_new.txt", "w+")
f5 = open("mapped_labels.txt", "r+")
list1 = []
list2 = []
list3 = []
prefix = "C:\\yolo\\darknet\\build\\darknet\\x64\\data\\obj_200\\"
#26K iterations
#list_map_label = ['tennisracket', 'pottedplant', 'banana', 'banana', 'banana', 'banana', 'donut', 'donut', 'banana', 'banana', 'donut', 'apple', 'apple', 'sandwich', 'banana', 'banana', 'banana', 'donut', 'banana', 'donut', 'donut', 'banana', 'tennisracket', 'keyboard', 'banana', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'tennisracket', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'person', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard', 'keyboard']
#10K iterations
#list_map_label = ['apple', 'banana', 'banana', 'banana', 'banana', 'banana', 'apple', 'banana', 'banana', 'banana', 'banana', 'apple', 'apple', 'banana', 'banana', 'banana', 'apple', 'donut', 'banana', 'banana', 'donut', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'person', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'banana', 'banana']
#5K iterations
list_map_label = ['banana', 'banana', 'banana', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'donut', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'apple', 'apple', 'apple', 'banana', 'apple', 'banana', 'tennisracket', 'tennisracket', 'donut', 'sandwich', 'tennisracket', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple']
#2.6K model
list_map_label = ['banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana', 'banana']
#my new model (7.6K)
#same as resnet 2.6K
#new_model only with 100 training samples
list_map_label = ['tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'tennisracket', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person', 'person']
new_list = []
for line in f1:
    temp = line.split(",")
    #new_temp = temp[-1].replace("_", "")
    #newest_temp = " ".join(new_temp.split())
    #list1.append(newest_temp)
    new_temp = " ".join(temp[-1].split())
    list1.append(new_temp)
    #print(newest_temp)
    #f3.write("%s\n"%newest_temp)
for line in f2:
    temp = line.split(",")
    #print(int(temp[1]))
    list2.append(int(temp[1]))

unique = []
for i in range(200):
    if list2[i] not in unique:
        unique.append(list2[i])

#print(unique)
print(len(unique))

count_absent = 0
for i in range(200, 249):
    #print(i)
    if list2[i] not in unique:
        print("\n missing = " + str(i))
        count_absent += 1

print(count_absent)

unique = []
for i in range(200, 249):
    if list2[i] not in unique:
        unique.append(list2[i])

#print(unique)
print(len(unique))


for line in f5:
    temp = " ".join(line.split())
    print(temp)
    list3.append(temp)


for i in range(len(list_map_label)):
    ind = list3.index(list_map_label[i])
    new_list.append(list1[ind])
    print(list1[i])

print(new_list)   
#"""
"""
for i in range(1,200):
    cmd_1 = "mv " + prefix + "pic_" + str(i) + "_org.png "
    cmd_2 = prefix + str(i) + "_" +list3[list2[i-1]] + ".png"
    cmd = cmd_1 + cmd_2
    print(i)
    f4.write("%s\n"%cmd_2)
    os.system(cmd)
"""
    