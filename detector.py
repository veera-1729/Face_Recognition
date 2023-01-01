import cv2
import numpy as np
from PIL import Image as im
import sqlite3
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cam = cv2.VideoCapture(0)
rec  = cv2.face_LBPHFaceRecognizer.create()
 
rec.read("recognizer\\trainingData.yml")
id=0
#font = cv2.cv.InitFont(cv2.cv.CV_Font_HERSHEY_COMPLEX_SMALL,5,1,0,4)#5 = size of font , 4=thickenss of font
font = cv2.FONT_HERSHEY_SIMPLEX

def getProfile(id):
    conn = sqlite3.connect("FaceBaseOf_People.db")
    cmd = "select * from FACES where ID="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    #print(type(profile))
    return profile

while(True):
    check,frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        id, conf  = rec.predict(gray[y:y+h,x:x+w])
        #cv2.putText(cv2.fromarray(frame),str(id),font ,(0,255,0))
        #cv2.putText(im.fromarray(frame),str(id),(10,10),font,1,(0,255,0),3,cv2.LINE_AA)
        profile = getProfile(id)
        if(profile != None):
            cv2.putText(frame,"ID:"+str(profile[0]),(x,y+h+30),font,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"Name:"+str(profile[1]),(x,y+h+60),font,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"Age:"+str(profile[2]),(x,y+h+90),font,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"Gender:"+str(profile[3]),(x,y+h+120),font,1,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow("Face",frame)
    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
