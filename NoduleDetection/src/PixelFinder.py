import random
import numpy as np
import pylab as pl
from Constants import MAX_FEAT_WINDOW

class PixelFinder: #TODO use threshold mask
    def __init__(self, xmlReader):
        self.Reader = xmlReader
        
    ############# NEW METHOD ####################
    # generate random x,y,z
    # get center,r from XmlAnnotationReader.GetNodulePosition()
    # check for every x,y,z whether the distance between x,y,z and every possible center is larger than e.g. 2r
    # if it is larger then store in NegativeList, otherwise store in PositiveList
    def findRandomNonNodulePixels(self, NumberPixelNeeded):
        # we generate a random number for x,y,z depending on the scandimensions
        maxXsize, maxYsize, maxZsize = self.Reader.dfr.getVolumeShape()
        
        while NumberPixelNeeded > 0:
            x = random.randint(MAX_FEAT_WINDOW,maxXsize-MAX_FEAT_WINDOW-1) #TODO find more elegant solution
            y = random.randint(MAX_FEAT_WINDOW,maxYsize-MAX_FEAT_WINDOW-1)
            z = random.randint(MAX_FEAT_WINDOW,maxZsize-MAX_FEAT_WINDOW-1)
            v = np.array([x,y])
            
            IsBadPixel = False
            #for c,r in reader.getNodulePositions(): #this only works if we use world coordinates, and the nodules are nice spheres
            for c,r in self.Reader.getNodulePositionsInSlice(z):
                if sum((v-c)**2) < (2*r)**2:
                    IsBadPixel = True
                    #print("Found a bad pixel at {},{},{}".format(x,y,z))
                    break
                
            if IsBadPixel: #find another random pixel
                continue
            
            NumberPixelNeeded -= 1
            
            yield x, y, z
    
    def findNodulePixels(self, method='circle', radiusFactor=1.0):
        m, n, _ = self.Reader.dfr.getVolumeShape()
        for nodule in self.Reader.Nodules:
            if method == 'circle':
                masks, _, _ = nodule.regions.getRegionMasksCircle(m,n, radiusFactor)
            elif method == 'polygon':
                _, masks = nodule.regions.getRegionMasksPolygon(m,n)
            else:
                raise ValueError('unsupported method')
            for z, mask in masks.iteritems():
                xs, ys = np.where(mask)
                for x, y in zip(xs, ys):
                    yield x, y, z
                    
    def plotHistograms(self):
        data = self.Reader.dfr.getVolumeData()
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