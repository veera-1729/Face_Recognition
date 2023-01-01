import cv2
import numpy as np
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cam = cv2.VideoCapture(0)
while(True):
    check,frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Face",frame)
    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
