import skvideo.io, cv2, time, os
srcfile = "pic_1_org.png"
outputfile = "encoding_try.mp4"
img = cv2.imread(srcfile, cv2.IMREAD_UNCHANGED)
writer = skvideo.io.FFmpegWriter(outputfile, outputdict={'-vcodec': 'libx264', '-qp': '25'})
writer.writeFrame(img)
writer.close()

start = time.time()
vidObj = cv2.VideoCapture(outputfile)
# Used as counter variable 
count = 0
print("\n time 1: ", (time.time()-start))

# checks whether frames were extracted 
success = 1
success, image = vidObj.read() 
print("\n time 2: ", (time.time()-start))
#inputdata = skvideo.io.vread(outputfile)
#cv2.imwrite("H:\\temp.jpg", image)
dim = (3840,2160)
# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite("temp.png", resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
print("\n time 3: ", (time.time()-start))

os.system("ffmpeg -i encoding_try.mp4 pic.jpg")
print("\n time 4: ", (time.time()-start))
os.system("ffmpeg -i pic.jpg -vf scale=3840:2160 pic.png")
print("\n time 5: ", (time.time()-start))
os.system("ffmpeg -i encoding_try.mp4 -vf scale=3840:2160 SPcheck.png")
print("\n time 6: ", (time.time()-start))