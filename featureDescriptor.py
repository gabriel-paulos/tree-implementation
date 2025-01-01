import numpy as np
import cv2 as cv

class FeatureDescriptor():

  def __init__(self,t="SIFT"):
    self.sift = cv.SIFT_create()
    self.mser = cv.MSER_create()
    self.t = t

  def extract_descriptors(self, img_path):

    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if self.t == "SIFT":
      kp, des = self.sift.detectAndCompute(img,None)
    
    elif self.t == "MSER":
      kp = self.mser.detect(img)   
      des = self.sift.compute(img, kp)  

    return kp, des


