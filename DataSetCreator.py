import cv2
import numpy as np
import sqlite3
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cam = cv2.VideoCapture(0)
def insertOrUpdate(Id,Name):
    conn = sqlite3.connect("FaceBaseOf_People.db")
    cmd = "select * from FACES where ID="+str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd = "update FACES  set Name="+str(Name)+" where ID="+str(Id)
    else:
        cmd = "insert into FACES (ID,Name)  Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
id = input("enter the id: ")
name = input("enter name: ")
insertOrUpdate(id,name)
SampleNum=0
while(True):

    check,frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray,1.3,5)
    #print(type(faces))
    for x,y,w,h in faces:
        SampleNum += 1
        cv2.imwrite("DataSet/sample."+id+"."+str(SampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)
    cv2.imshow("Face",frame)
    cv2.waitKey(1)
    if(SampleNum>10):
        break
cam.release()
cv2.destroyAllWindows()
