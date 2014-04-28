import numpy as np
from FeatureGenerator import FeatureGenerator

class Classifier:
    def __init__(self, data, vshape):
        self.Data = data
        self.VoxelShape = vshape
        self.clf = None
        self.fgen = None
        
    def __del__(self):
        del self.Data
        del self.VoxelShape
        del self.clf
        del self.fgen
    
    def isLevelset(self):
        return self.fgen is not None and self.clf is not None
    
    def setLevel(self, level, clf):
        self.fgen = FeatureGenerator(self.Data, self.VoxelShape, level)
        self.clf = clf
        
#     @staticmethod
#     def generatePixelList2D((h,w)):
#         x, y = np.meshgrid(np.arange(h), np.arange(w))
#         x, y = x.flatten(), y.flatten()
#         points = np.vstack((x,y)).T
#         assert points.shape == (h*w,2)
#         
#         del x,y
#         return points
#     
#     @staticmethod
#     def generatePixelList3D((h,w,d)):
#         x, y, z = np.meshgrid(np.arange(h), np.arange(w), np.arange(d))
#         x, y, z = x.flatten(), y.flatten(), z.flatten()
#         points = np.vstack((x,y,z)).T
#         assert points.shape == (h*w*d,3)
#         
#         del x,y,z
#         return points
    
    def generateProbabilityVolume(self, mask3D, threshold=0.01):
        if not self.isLevelset():
            raise ValueError("Level not set")
        
        testFeatures = self.fgen.getAllFeaturesByMask(mask3D)
        
        m,n = testFeatures.shape
        maxChunk = 1000000
        nbRows = maxChunk // n
        
        result = np.empty((m,2))
        for r in np.arange(0,m,nbRows):
            print "[{}, {}[".format(r, r+nbRows)
            chunk = testFeatures[r:r+nbRows,:]
            result[r:r+nbRows,:] = self.clf.predict_proba(chunk)
        
        #result = self.clf.predict_proba(testFeatures)
        
        probImg = np.zeros(mask3D.shape)
        probImg[mask3D] = result[:,1]
        
        mask = probImg > threshold
        
        return probImg, mask
    
#     def generateProbabilityImage(self, mask2D, mySlice, threshold=0.01):
#         if not self.isLevelset():
#             raise ValueError("Level not set")
#         
#         testFeatures = deque()
#         for px,py in zip(np.where(mask2D)):
#             pixelFeatures = self.fgen.calculatePixelFeatures(px, py, mySlice)
#             testFeatures.append(pixelFeatures)
#     
#         testFeatures = np.array(testFeatures)
#         result = self.clf.predict_proba(testFeatures)
#         
#         probImg = np.zeros(mask2D.shape)
#         probImg[mask2D] = result[:, 1]
#         
#         mask = ma.masked_greater(probImg, threshold).mask
#         
#         return probImg, mask