import operator 
# Declaring initial lists 
list1 = ['akshat', 'Manjeet', 'nikhil'] 
list2 = [0.55, 0.72, 0.3] 
zipped = zip(list1, list2) 
  
# Converting to list 
zipped = list(zipped) 
  
# Printing zipped list 
print("Initial zipped list - ", str(zipped)) 
  
# Using sorted and operator 
res = sorted(zipped, key = operator.itemgetter(1)) 
      
# printing result 
print("final list - ", str(res)) 


res1 = list(zip(*res)) 
      
# Printing modified list  
print ("Modified list is : " + str(res1))
 
print(res1[0])
print(res1[1])

out = [item for item in res1[0]] 
  
# printing output 
out.reverse()
print(out) 