import cv2, time, os
import skvideo.io, torch

'''
persist :image: object to disk. if path is given, load() first.
jpg_quality: for jpeg only. 0 - 100 (higher means better). Default is 95.
png_compression: For png only. 0 - 9 (higher means a smaller size and longer compression time).
              Default is 3.
'''

#please use time.clock() or cv2.getTickCount() for measurement.
#you also should restrict it to the processing, not imread() imshow(), etc.
#timing analysis
"""
opencv: time to read the image =  360.3930473327637
opencv: time to resize and save and jpeg compress =  439.3191337585449
ffmpeg: time to resize and jpeg compress=  898.0309963226318
ffmpeg: time to combined resize and jpeg compress=  408.7414741516113
ffmpeg: time to encode=  812.5317096710205
ffmpeg: time to encode on resized and jpeg compressed=  194.88859176635742
"""

#"""
srcfile = "H:\\drone_video_cp_11\\pic_2_org.png"
cv_destfile = "H:\\cv_downsize.png"
cv_destfile_jpg = "H:\\cv_downsize_jpg.jpg"
ffmpeg_destfile = "H:\\ffmpeg_downsize.png"
ffmpeg_destfile_jpg = "H:\\ffmpeg_downsize_jpg.jpg"
ffmpeg_destfile_combined = "H:\\ffmpeg_downsize_jpg_combo.jpg"
encode_file = "H:\\ffmpeg_encode.mp4"
outputfile = "H:\\skvideo_encode.mp4"

image_list = []
for i in range(1,4):
    srcfile  = "H:\\drone_video_cp_11\\pic_" + str(i) +"_org.png"
    img = cv2.imread(srcfile, cv2.IMREAD_UNCHANGED)
    image_list.append(img)


#replace ffmpeg command line utility to cv2
#reading the image takes most of the time (preread all training images beforehand) --> 50% of total time
#can we use GPU acceleration for opencv?
#can PIL be an alternative?

start = time.time()
#img = cv2.imread(srcfile, cv2.IMREAD_UNCHANGED)
img = image_list[1]
duration_read = (time.time() - start)*1000
 
print('Original Dimensions : ',img.shape)
 
scale_percent = 80 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
dim = (963, 541)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
"""
#for JPEG compression
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
result, encimg = cv2.imencode('.jpg', resized, encode_param)
print(encimg)
cv2.imwrite("H:\\cv2_jpeg_try.jpg",cv2.imdecode(encimg, cv2.IMREAD_COLOR))
"""
#The default value for IMWRITE_JPEG_QUALITY is 95, range is 0-100
#cv2.imwrite(cv_destfile,resized)
cv2.imwrite(cv_destfile_jpg, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#cv2.imwrite(cv_destfile_jpg,encimg)
duration_cv = (time.time() - start)*1000
#skvideo encoding
start = time.time()
writer = skvideo.io.FFmpegWriter(outputfile, outputdict={'-vcodec': 'libx264', '-qp': '25'})
print("\n time1: ", (time.time() - start))
img = cv2.imread(cv_destfile_jpg, cv2.IMREAD_UNCHANGED)
print("\n time2: ", (time.time() - start))
writer.writeFrame(img)
print("\n time3: ", (time.time() - start))
writer.close()
#writer = skvideo.io.FFmpegWriter("temp2.mp4", outputdict={'-vcodec': 'libx264', '-qp': '25'})
print("\n time4: ", (time.time() - start))
duration_sk_mp4 = (time.time() - start)*1000
start = time.time()
cmd_str = "ffmpeg -i " + cv_destfile_jpg +" -c:v libx264 -qp 25 H:\\temp.mp4"
os.system(cmd_str)
print("\n time5: ", (time.time() - start))
#"""
"""
while True:
    print()
"""
"""
#ffmpeg part
start = time.time()
cmd_str = "ffmpeg -i " + srcfile +" -vf scale=" + str(width)+":" + str(height) + " " + ffmpeg_destfile
os.system(cmd_str)
cmd_str = "ffmpeg -i " + ffmpeg_destfile +" -q:v 90 "+ ffmpeg_destfile_jpg
os.system(cmd_str) 
duration_ffmpeg = (time.time() - start)*1000
#ffmpeg combined downsizing and jpeg compression (best working)
start = time.time()
cmd_str = "ffmpeg -i " + srcfile +" -vf scale=" + str(width)+":" + str(height) + " -q:v 90 " + ffmpeg_destfile_combined
os.system(cmd_str)
duration_ffmpeg_combined = (time.time() - start)*1000
#encoding orginal final
start = time.time()
cmd_str = "ffmpeg -i " + srcfile +" -c:v libx264 -qp 25 " + encode_file
os.system(cmd_str)
duration_ffmpeg_mp4 = (time.time() - start)*1000
#encoding orginal final
start = time.time()
#cmd_str = "ffmpeg -i " + ffmpeg_destfile_combined +" -c:v libx264 -qp 25 H:\\temp.mp4"
cmd_str = "ffmpeg -i " + cv_destfile_jpg +" -c:v libx264 -qp 25 H:\\temp.mp4"
os.system(cmd_str)
duration_ffmpeg_mp4_reduced = (time.time() - start)*1000
start = time.time()
cmd_str = "ffmpeg -i " + srcfile +" -vf scale=" + str(width)+":" + str(height) + " -q:v 90 -c:v libx264 -qp 25 H:\\temp_combo.mp4"
os.system(cmd_str)
duration_ffmpeg_mp4_combo = (time.time() - start)*1000

print("\n opencv: time to read the image = ", duration_read)
print("\n opencv: time to resize and save and jpeg compress = ", duration_cv)
print("\n sk_video: time to only encode = ", duration_sk_mp4)
print("\n ffmpeg: time to resize and jpeg compress= ", duration_ffmpeg)
print("\n ffmpeg: time to combined resize and jpeg compress= ", duration_ffmpeg_combined)
print("\n ffmpeg: time to encode= ", duration_ffmpeg_mp4)
print("\n ffmpeg: time to encode on resized and jpeg compressed= ", duration_ffmpeg_mp4_reduced)
print("\n ffmpeg: time to encode on resized and jpeg combo= ", duration_ffmpeg_mp4_combo)
"""
"""
y=[]
x = torch.tensor(1.0, requires_grad = True)
y.append(x)
x = torch.tensor(1.0, requires_grad = True)
y.append(x)
x = torch.tensor(1.0, requires_grad = True)
y.append(x)
x = torch.tensor(1.0, requires_grad = True)
y.append(x)
print(y)
print(sum(y))
"""
