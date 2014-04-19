import random
import numpy as np
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
                _, masks = nodule.regions.getRegionMasksPolygon(m,n) #TODO switch back?
            else:
                raise ValueError('unsupported method')
            for z, mask in masks.iteritems():
                xs, ys = np.where(mask)
                for x, y in zip(xs, ys):
                    yield x, y, z

# from XmlAnnotationReader import XmlAnnotationReader
# myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
# #myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
# reader = XmlAnnotationReader(myPath)    
# finder = PixelFinder(reader)
# 
# pixels = list(finder.findNodulePixels())
# print("Found {} nodule pixels".format(len(pixels)))
# for x,y,z in pixels:
#     print x,y,z
    
#for x,y,z in finder.findRandomNonNodulePixels(500):
#    print(x,y,z)

############## STORAGE ########################        
# store trainingsdata for further use
# import pickle
# pixelTraining001 = pixelTraining # give specific name to trainingsset
# f = open('pixelTraining_LIDC001.pkl', 'wb') # give name to document
# pickle.dump(pixelTraining001, f, pickle.HIGHEST_PROTOCOL)
# f.close()

######################## FOR LARGE DATASETS
# import tables
# h5file = tables.openFile('test.h5', mode='w', title="Test Array")
# root = h5file.root
# h5file.createArray(root, "test", a)
# h5file.close()     
                 
######################## OPEN pickle
#pixelTraining2 = pickle.load( open( 'pixelTraining_LIDC001.pkl', 'rb') )
#print(np.array_equal(pixelTraining2, pixelTraining001))

