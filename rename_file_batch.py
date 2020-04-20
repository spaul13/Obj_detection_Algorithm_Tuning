from os import listdir
from os.path import isfile, join
import sys
import os
mypath = sys.argv[1]
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for files in onlyfiles:
    cmd_str = "mv " + sys.argv[1] + files + " " + sys.argv[1] + files[:-4] + "_org.png"
    print(cmd_str)
    os.system(cmd_str)

print(onlyfiles)