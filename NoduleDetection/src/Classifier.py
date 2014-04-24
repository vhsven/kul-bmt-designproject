import numpy as np
import numpy.ma as ma
from collections import deque
from DicomFolderReader import DicomFolderReader
from FeatureGenerator import FeatureGenerator
#from msilib import Feature

class Classifier:
    def __init__(self, myPath):
        self.dfr = DicomFolderReader(myPath)
        self.clf = None
        self.fgen = None
    
    def isLevelset(self):
        return self.fgen is not None and self.clf is not None
    
    def setLevel(self, level, clf):
        vData = self.dfr.getVolumeData()
        voxelShape = self.dfr.getVoxelShape()
        self.fgen = FeatureGenerator(vData, voxelShape, level)
        self.clf = clf
        
    @staticmethod
    def generatePixelList2D((h,w)):
        x, y = np.meshgrid(np.arange(h), np.arange(w))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T
        assert points.shape == (h*w,2)
        
        del x,y
        return points
    
    @staticmethod
    def generatePixelList3D((h,w,d)):
        x, y, z = np.meshgrid(np.arange(h), np.arange(w), np.arange(d))
        x, y, z = x.flatten(), y.flatten(), z.flatten()
        points = np.vstack((x,y,z)).T
        assert points.shape == (h*w*d,3)
        
        del x,y,z
        return points
    
    def generateProbabilityVolume(self, mask3D, threshold=0.01):
        if not self.isLevelset():
            raise ValueError("Level not set")
        
        testFeatures = self.fgen.getAllFeatures(mask3D)
        result = self.clf.predict_proba(testFeatures)
        
        probImg = np.zeros(mask3D.shape)
        #probImg[points[:,0], points[:,1], points[:,2]] = result[:,1]
        probImg[mask3D] = result[:,1]
        
        mask = ma.masked_greater(probImg, threshold).mask
        
        return probImg, mask
    
    def generateProbabilityImage(self, mask2D, mySlice, threshold=0.01):
        if not self.isLevelset():
            raise ValueError("Level not set")
        
        testFeatures = deque()
        for px,py in zip(np.where(mask2D)):
            pixelFeatures = self.fgen.calculatePixelFeatures(px, py, mySlice)
            testFeatures.append(pixelFeatures)
    
        testFeatures = np.array(testFeatures)
        result = self.clf.predict_proba(testFeatures)
        
        probImg = np.zeros(mask2D.shape)
        #probImg[points[:,0], points[:,1]] = result[:,1]
        probImg[mask2D] = result[:, 1]
        
        mask = ma.masked_greater(probImg, threshold).mask
        
        return probImg, mask