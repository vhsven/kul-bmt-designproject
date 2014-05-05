import sys
import scipy.stats
import numpy as np
import math
import collections
from collections import deque
from skimage.filter.rank import entropy
from skimage.morphology import disk
import scipy.ndimage as nd
import scipy.ndimage.morphology as morph
from scipy.ndimage.filters import generic_gradient_magnitude, sobel
from Preprocessor import Preprocessor
from numpy import argwhere

class FeatureGenerator:
    def __init__(self, setID, data, vshape, level=1):
        self.SetID = setID
        self.Data = data
        self.VoxelShape = vshape
        self.Level = level
        self.PixelCount = None
        
    def __str__(self):
        return "Level {} Feature Generator".format(self.Level)
    
    def __del__(self):
        del self.SetID
        del self.Data
        del self.VoxelShape
        del self.Level
        del self.PixelCount
        
    def getSlice(self, z):
        return self.Data[:,:,int(z)]
    
    @staticmethod
    def getWindowFunctionByMask(mask3D, f, windowSize=3):
        xs,ys,zs = np.where(mask3D)
        coords = zip(xs,ys,zs)
        count = len(coords)
        x,y,z = coords[0]
        first = f(x, y, z, windowSize)
        if hasattr(first,"__len__"):
            leng = len(first)
        else:
            leng = 1
            
        result = np.zeros((count, leng))
        result[0,:] = first
        for i in range(1, count):
            x,y,z = coords[i]
            result[i,:] = f(x, y, z, windowSize) 
        
        return result
    
    # N: number of datapoints
    # l(i): total features per level i
    # L: total features over all levels
    def getLevelFeatureByMask(self, level, mask3D):
        """Returns a ndarray (Nxl) containing the features for all given positions and for the given level.""" 
        if level == 1:
            return self.getIntensityByMask(mask3D)
        if level == 3:
            N = mask3D.sum()
            start,stop = 2,8
            result = np.empty((N,stop-start+1))
            for sigma in np.arange(start,stop):
                sigmas = np.array([sigma]*3) / np.array(self.VoxelShape)
                print(sigmas)
                result[:,sigma-start] = self.getLaplacianByMask(mask3D, sigmas)
            result[:,stop-start] = self.getEdgeDistByMask(mask3D, self.SetID, sigma=4.5)
            return result
        if level == 2:
            return FeatureGenerator.getWindowFunctionByMask(mask3D, self.averaging3D)
        if level == 4:
            return FeatureGenerator.getWindowFunctionByMask(mask3D, self.getStats)
        else:
            print("Falling back on features per pixel method.")
            result = deque()
            for x,y,z in zip(np.where(mask3D)):
                feature = self.getLevelFeature(level, x, y, z)
                result.append(feature)
            return np.array(result).reshape((-1, 1))
        
    def getAllFeaturesByMask(self, mask3D):
        """Returns a ndarray (NxL) with the rows containing the features vector, up to the current level, per datapoint."""
        nbVoxels = mask3D.sum()
        h,w,d = mask3D.shape
        totalVoxels = h*w*d
        ratio = 100.0 * nbVoxels / totalVoxels
        print("Generating features for {0} ({1:.2f}%) voxels.".format(nbVoxels, ratio))
        allFeatures = self.getLevelFeatureByMask(1, mask3D)
        for level in range(2, self.Level+1):
            lvlFeature = self.getLevelFeatureByMask(level, mask3D) #Nxl
            allFeatures = np.hstack([allFeatures, lvlFeature])
        
        return allFeatures
    
#     def getLevelFeature(self, level, x,y,z):
#         """Returns a scalar representing the feature at the given position for the given level."""
#         if level == 1:
#             return self.getIntensity(x, y, z)
#         if level == 2: #TODO this is very inefficient
#             start,stop = 2,6
#             result = np.empty((1,stop-start+1))
#             for sigma in np.arange(start,stop):
#                 sigmas = np.array([sigma]*3) / np.array(self.VoxelShape)
#                 print sigmas
#                 result[:,sigma-start] = self.getLaplacian(x,y,z, sigmas)
#             result[:,stop-start] = self.getBlurredEdges(x,y,z, self.SetID, sigma=4.5)
#             return result
#         #if level == 3:
#         #    return self.getEntropy(x,y,z, windowSize=5)
#         #if level == 4:
#         #    return self.getEdges(x, y, z)
#         else:
#             raise ValueError("Level {} not supported.".format(level))
#     
#     def getAllFeatures(self, x,y,z): #TODO distance from lung wall?
#         """Returns a ndarray (1xL) containing the feature vector, up to the current level, for the given datapoint."""
#         z = int(z)
#         
#         allFeatures = self.getLevelFeature(1, x, y, z)
#         #allFeatures = deque()
#         for level in range(2, self.Level+1):
#             lvlFeature = self.getLevelFeature(level, x, y, z)
#             allFeatures = np.hstack([allFeatures, lvlFeature])
#             #allFeatures.append(lvlFeature)
#         
#         return np.array(allFeatures).reshape((1,-1))
# 
#     def getIntensity(self, x, y, z):
#         """Returns a scalar representing the intensity at the given position."""
#         return self.Data[x,y,z]
    
    def getIntensityByMask(self, mask3D):
        """Returns an array (Nx1) containing the intensities of all the given positions."""
        intensities = self.Data[mask3D]
        return intensities.reshape((-1, 1))
    
#     def getRelativePosition(self, x, y, z):
#         h,w,d = self.Data.shape
#         return float(x)/h, float(y)/w, float(z)/d
#     
#     def getRelativePositionByMask(self, mask3D):
#         h,w,d = self.Data.shape
#         xs, ys, zs = np.where(mask3D)
#         xsr = xs / float(h)
#         ysr = ys / float(w)
#         zsr = zs / float(d)
#         return np.vstack([xsr,ysr,zsr]).T
    
    def getLaplacianByMask(self, mask3D, sigmas):
        B = argwhere(mask3D)
        (xstart, ystart, zstart), (xstop, ystop, zstop) = B.min(0)-2, B.max(0)+3 
        data = self.Data[xstart:xstop, ystart:ystop, zstart:zstop]
        mask = mask3D[xstart:xstop, ystart:ystop, zstart:zstop]
        return nd.filters.gaussian_laplace(data, sigmas)[mask]
            
    def getEdgeDistByMask(self, mask3D, setID, sigma=4.5):
        result = Preprocessor.loadThresholdMask(setID)
        #result = generic_gradient_magnitude(result, sobel).astype(np.float32)
        #result = nd.filters.gaussian_filter(result, sigma)
        result = morph.distance_transform_cdt(result, metric='taxicab').astype(np.float32)
        return result[mask3D]
    
    ############################################################
    # 3D averaging (Keshani et al.)
    ############################################################
#     def averaging3DByMask(self, mask3D, windowSize=3):
#         xs,ys,zs = np.where(mask3D)
#         coords = zip(xs,ys,zs)
#         count = len(coords)
#         result = np.zeros((count, 1))
#         for i in range(0, count):
#             x,y,z = coords[i]
#             result[i,0] = self.averaging3D(x, y, z, windowSize) 
#         
#         return result
    
    def averaging3D (self, x,y,z, windowSize=3):
        # square windowSize x windowSize
        valdown = windowSize // 2
        valup   = valdown + 1
        
        # nodules will continue in preceeding/succeeding slices but bronchioles will not
        # assume: nodules have minimum length of 5 mm
        Q = int(5 // self.VoxelShape[2]) # = c / T = 5mm / thickness of slices
    
        def getWindowMean(p):
            if p >= self.Data.shape[2]:
                print("Value p={} is too large.".format(p)) #TODO fix
                return getWindowMean(p-1) #Idee van Kim
            
            windowP = self.Data[x-valdown:x+valup,y-valdown:y+valup,p]
            return windowP.mean()
        
        meanMin = 0
        for p in range(z-Q, z):
            meanMin += getWindowMean(p)
        meanPlus = 0
        for p in range(z+1, z+Q+1):
            meanPlus += getWindowMean(p)
        
        return (meanMin/Q) * (meanPlus/Q)
        
    def getStats(self, x,y,z, windowSize=3):
        valdown = windowSize // 2
        valup   = valdown + 1
        
        windowD=self.Data[x-valdown:x+valup,y-valdown:y+valup,z-valdown:z+valup]
        
        #skewness
        skx = scipy.stats.skew(windowD, axis=0).mean()
        sky = scipy.stats.skew(windowD, axis=1).mean()
        skz = scipy.stats.skew(windowD, axis=2).mean()
        
        #Kurtosis
        kx = scipy.stats.kurtosis(windowD, axis=0).mean()
        ky = scipy.stats.kurtosis(windowD, axis=1).mean()
        kz = scipy.stats.kurtosis(windowD, axis=2).mean()
        
        return np.array([skx,sky,skz,kx,ky,kz])
    
    ############################################################
    # feature[6]= sliceEntropy calculation (disk window or entire image)
    ############################################################        
    def getEntropyByMask(self, mask3D, windowSize):
        sys.stdout.write("Calculating entropy")
        _,_,d = self.Data.shape
        returnValue = np.array([])
        for z in range(0,d):
            sys.stdout.write('.')
            mySlice = self.Data[:,:,z].astype('uint8')
            #mySlice = img_as_ubyte(mySlice)
            mask = mask3D[:,:,z]
            entropySlice = entropy(mySlice, disk(windowSize))
            result = entropySlice[mask]
            returnValue = np.append(returnValue, result)
        
        print("")
        nbValues = len(returnValue)
        return returnValue.reshape(nbValues, 1)
        #if windowSize not in self.Entropy.keys():
        #    data8 = self.Data.astype('uint8')
        #    self.Entropy[windowSize] = entropy(data8, ball(windowSize))
        
        #return self.Entropy[windowSize][mask3D].T
    
    def image_entropy(self, z):
        # calculates the sliceEntropy of the entire slice
        img=self.getSlice(z)
        histogram,_ = np.histogram(img,100)
        histogram_length = sum(histogram)
    
        samples_probability = [float(h) / histogram_length for h in histogram]
        image_entropy=-sum([p * math.log(p, 2) for p in samples_probability if p != 0])
    
        return image_entropy
