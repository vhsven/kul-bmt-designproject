import random
import numpy as np
import pylab as pl
from Constants import MAX_FEAT_WINDOW
from XmlAnnotationReader import XmlAnnotationReader

class PixelFinder:
    def __init__(self, myPath, cc):
        self.Reader = XmlAnnotationReader(myPath, cc)
        
    def __del__(self):
        del self.Reader
    
    def getLists(self, shape, method='circle', radiusFactor=1.0):
        #print("\tSearching for nodule pixels.")
        pixelCacheP = list(self.findNodulePixels(shape, method, radiusFactor))
            
        #print("\tSearching for non-nodule pixels.")
        nbPixels = len(pixelCacheP)
        pixelCacheN = list(self.findRandomNonNodulePixels(shape, nbPixels))
            
        return pixelCacheP, pixelCacheN
    
    def getMasks(self, shape, method='circle', radiusFactor=1.0):
        
        maskP = self.findNodulePixelsMask(shape, method, radiusFactor)
        
        nbNeeded = maskP.sum()    
        maskN = self.findRandomNonNodulePixelsMask(shape, nbNeeded)
            
        return maskP, maskN, nbNeeded
        
        
    ############# NEW METHOD ####################
    # generate random x,y,z
    # get center,r from XmlAnnotationReader.GetNodulePosition()
    # check for every x,y,z whether the distance between x,y,z and every possible center is larger than e.g. 2r
    # if it is larger then store in NegativeList, otherwise store in PositiveList
    def findRandomNonNodulePixels(self, shape, nbNeeded):
        # we generate a random number for x,y,z depending on the scandimensions
        maxXsize, maxYsize, maxZsize = shape
        
        nbIterations = 0
        while nbNeeded > 0:
            nbIterations += 1
            if nbIterations > 10000:
                raise Exception("Can't find enough good random pixels in volume {}x{}x{}.".format(maxXsize, maxYsize, maxZsize))
            x = random.randint(MAX_FEAT_WINDOW,maxXsize-MAX_FEAT_WINDOW-1)
            y = random.randint(MAX_FEAT_WINDOW,maxYsize-MAX_FEAT_WINDOW-1)
            z = random.randint(MAX_FEAT_WINDOW,maxZsize-MAX_FEAT_WINDOW-1)
            
            p = np.array([x,y,z])
            IsBadPixel = False
            for nodule in self.Reader.Nodules:
                if nodule.Regions.isPointInsideCircles(p, max, radiusFactor=2.0):
                    IsBadPixel = True
                    break
                
#             v = np.array([x,y])
#             for c,r in self.Reader.getNodulePositionsInSlice(z, max):
#                 if sum((v-c)**2) < (2*r)**2:
#                     IsBadPixel = True
#                     #print("Found a bad pixel at {},{},{}".format(x,y,z))
#                     break
                
            if IsBadPixel: #find another random pixel
                continue
            
            nbIterations = 0
            nbNeeded -= 1
            
            yield x, y, z
            
    def findRandomNonNodulePixelsMask(self, shape, nbNeeded):
        mask3D = np.zeros(shape, dtype=np.bool)
        
        for x,y,z in self.findRandomNonNodulePixels(shape, nbNeeded):
            mask3D[x,y,z] = True
            
        return mask3D
    
    def findNodulePixels(self, shape, method='circle', radiusFactor=1.0):
        m,n,_ = shape
        for nodule in self.Reader.Nodules:
            if method == 'circle':
                masks, _, _ = nodule.Regions.getRegionMasksCircle(m, n, min, radiusFactor)
            elif method == 'polygon':
                _, masks = nodule.Regions.getRegionMasksPolygon(m,n)
            else:
                raise ValueError('unsupported method')
            for z, mask in masks.iteritems():
                xs, ys = np.where(mask)
                for x, y in zip(xs, ys):
                    yield x, y, z
                    
    def findNodulePixelsMask(self, shape, method='circle', radiusFactor=1.0):
        mask3D = np.zeros(shape, dtype=np.bool)
        m,n,_ = shape
        for nodule in self.Reader.Nodules:
            if method == 'circle':
                masks, _, _ = nodule.Regions.getRegionMasksCircle(m,n, min, radiusFactor)
            elif method == 'polygon':
                _, masks = nodule.Regions.getRegionMasksPolygon(m,n)
            else:
                raise ValueError('unsupported method')
            for z, mask in masks.iteritems():
                mask3D[:,:,int(z)] = np.bitwise_or(mask, mask3D[:,:,int(z)]) 
                
        return mask3D
                    
    def plotHistograms(self, data):
        pixelsP = list(self.findNodulePixels(radiusFactor=0.33))
        pixelsN = list(self.findRandomNonNodulePixels(len(pixelsP)))
        
        xsp,ysp,zsp = zip(*pixelsP)
        xsn,ysn,zsn = zip(*pixelsN)
        
        intensitiesP = data[xsp,ysp,zsp]
        intensitiesN = data[xsn,ysn,zsn]
        
        pl.subplot(121)
        pl.hist(intensitiesP, 10)
        pl.title('Histogram of positive (nodule) pixels')
        pl.subplot(122)
        pl.hist(intensitiesN, 10)
        pl.title('Histogram of negative (non-nodule) pixels')
        pl.show()

#from DicomFolderReader import DicomFolderReader
#dfr = DicomFolderReader(myPath)
#data = dfr.getVolumeData()   
#finder = PixelFinder(myPath)
#finder.plotHistograms(data)