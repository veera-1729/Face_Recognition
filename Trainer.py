import os
import cv2
import numpy as np
from PIL import Image

recognizer= cv2.face.LBPHFaceRecognizer_create(	)

path = 'DataSet'

def getImagesWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    print(imagePaths)

getImagesWithID(path)


