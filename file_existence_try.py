import os, os.path
fn = "file_size_log_0417_2.txt"
print(os.path.exists(fn))
if(os.path.exists(fn)):
	print("\n exists \n")
else:
	print("not exist\n")