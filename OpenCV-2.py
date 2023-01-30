import cv2 #opencv
import time #dealy
import imutils

cam = cv2.VideoCapture(0) #initializing camera
time.sleep(1) #dealy for 1 second

firstFrame = None #initial frame in camera
area = 500

while True:
    _, img = cam.read() #read frame from camera
    text = 'Normal'
    img = imutils.resize(img, width=500) #resize

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #gray scalling image

    gaussianimg = cv2.GaussianBlur(grayimg, (21, 21), 0) #blurring the gray image so no sharp edges

    if firstFrame is None:
        firstFrame = gaussianimg #capturing first frame on 1st iteration
        continue
    imgDiff = cv2.absdiff(firstFrame, gaussianimg) #absolute diff between the first frame with no obejct and then the new frame with object  
    threshimg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] #binary 
    threshimg = cv2.dilate(threshimg, None, iterations = 2) #to ignore minor differences
    contours = cv2.findContours(threshimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    #contours to recognize new object

    for c in contours:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = 'Moving Object Detected'

    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('cameraFeed', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  #camera will be relased and camera window will be destroyed by pressing q in keyboard
        break
cam.release()
cv2.destroyAllWindows() #to tackle any error
