import numpy as np
import numpy.ma as ma #TODO work without masks?
from collections import deque
from DicomFolderReader import DicomFolderReader
from FeatureGenerator import FeatureGenerator
from msilib import Feature

class Classifier:
    def __init__(self, myPath, clf, level=1):
        self.dfr = DicomFolderReader(myPath)
        self.setLevel(level, clf)
    
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
    
    def generateProbabilityVolume(self, shape, points):
        testFeatures = deque()
        for px,py,pz in points:
            pixelFeatures = self.fgen.calculatePixelFeatures(px, py, pz)
            testFeatures.append(pixelFeatures)
    
        testFeatures = np.array(testFeatures)
        result = self.clf.predict_proba(testFeatures)
        
        probImg = np.zeros(shape)
        probImg[points[:,0], points[:,1], points[:,2]] = result[:,1]
        
        masked = ma.masked_greater(probImg, 0.01)
        
        #TODO return list of points for use in next cascade
        
        return probImg, masked
    
    def generateProbabilityImage(self, shape, points, mySlice):    
        testFeatures = deque()
        for px,py in points:
            pixelFeatures = self.fgen.calculatePixelFeatures(px, py, mySlice)
            testFeatures.append(pixelFeatures)
    
        testFeatures = np.array(testFeatures)
        result = self.clf.predict_proba(testFeatures)
        
        probImg = np.zeros(shape)
        probImg[points[:,0], points[:,1]] = result[:,1]
        
        masked = ma.masked_greater(probImg, 0.01)
        
        return probImg, masked