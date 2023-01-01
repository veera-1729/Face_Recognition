import os
import cv2
import numpy as np
from PIL import Image

recognizer= cv2.face_LBPHFaceRecognizer.create()

path = 'DataSet'

def getImagesWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #print(imagePaths)
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8') #unsigned integer 0 t0 255
        ID = int(os.path.split(imagePath)[-1].split('.')[1]) #split the name of the image to get id init.
        faces.append(faceNp) #store the face as numpy array
        IDs.append(ID) 
        cv2.imshow("training",faceNp)
        cv2.waitKey(100)
    return faces, IDs
faces,IDs = getImagesWithID(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()
    




