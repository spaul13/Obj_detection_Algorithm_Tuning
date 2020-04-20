import subprocess, time, cv2, os
from ffmpy import FFmpeg
from PIL import Image
infile = "H:\\drone_video_cp_11\\pic_1_org.png"
"""
ff = FFmpeg(inputs={'pipe:0': '-f rawvideo -pix_fmt rgb24 -s:v 640x480'},outputs={'pipe:1': '-c:v h264 -f mp4'})
print(ff.cmd)
inputdata=open(infile, 'rb').read()
stdout, stderr = ff.run(input_data=inputdata, stdout=subprocess.PIPE)
"""
"""
start = time.time()
foo = Image.open(infile)
print("\n PIL:time taken to open the file= %f"%(time.time() - start)*1000)
foo = foo.resize((960,540),Image.ANTIALIAS)
print("\n PIL:time taken to resize the file= %f"%(time.time() - start)*1000)
foo.save("pil_temp.jpg",quality=95)
print("\n PIL:time taken to save the file= %f\n =========\n"%(time.time() - start)*1000)
# The saved downsized image size is 24.8kb
#foo.save("path\\to\\save\\image_scaled_opt.jpg",optimize=True,quality=95)
#opencv part
start = time.time()
img = cv2.imread(infile, cv2.IMREAD_UNCHANGED)
print("\n CV: time taken to open the file= %f"%(time.time() - start)*1000)
resized = cv2.resize(img, (960,540), interpolation = cv2.INTER_AREA)
print("\n CV: time taken to resize the file= %f"%(time.time() - start)*1000)
cv2.imwrite("cv_temp.jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
print("\n CV: time taken to save the file= %f"%(time.time() - start)*1000)
"""
#decoding using pyav, pil
"""
import av
v = av.open('path/to/video.mov')
for packet in container.demux():
    for frame in packet.decode():
        if frame.type == 'video':
            img = frame.to_image()  # PIL/Pillow image
            arr = np.asarray(img)  # numpy array
            # Do something!
"""
start = time.time()
os.system("ffmpeg -i pic_1_org.png -vf scale=960:-1 -q:v 80 -c:v libx264 -qp 25 -y temp_1.mp4")
print("\n time needed to execute : %f ms"%((time.time() - start)*1000))


