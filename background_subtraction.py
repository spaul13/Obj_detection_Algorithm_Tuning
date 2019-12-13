import numpy as np
import cv2

videofile = "F:\decoded_jockey\jockey_bg_20.mp4"
#videofile = "F:\decoded_shark_4K\raw_frame\shark_bg.mp4"
videofile = "F:\drone_video_cp_2\temp\pic_55_encoded.mp4"

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(videofile)

# Create the kernel that will be used to remove the noise in the foreground mask (additional)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

fgbg = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=1000, detectShadows=0)

#fgbg = cv2.BackgroundSubtractorMOG()

#fgbg = cv2.createbackgroundSubstractor()

#fgbg = cv2.BackgroundSubstractor()


i = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    # Obtain the foreground mask
    fgmask = fgbg.apply(frame)
    
    #Remove part of the noise(additional)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    print("here")
    cv2.imshow('frame',fgmask)
    cv2.imwrite('F:\drone_video_cp_2\temp\kang_'+str(i)+'_org.png',frame)
    i+=1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()