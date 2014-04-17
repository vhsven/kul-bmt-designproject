'''
Created on 12-apr.-2014

@author: Eigenaar
'''
import scipy
import scipy.ndimage
import numpy as np
import pylab
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from featureselection import FeatureSelection
from XmlAnnotationReader import XmlAnnotationReader
from Constants import *

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
reader = XmlAnnotationReader(myPath)
data = reader.dfr.getVolumeData()
select = FeatureSelection(data, reader.dfr.getVoxelShape())

####################################################
# training classifier
####################################################

# take slices in which a nodule is identified
pixelTraining = []
for nodule in reader.Nodules:
    masks, centerMap, r2 = nodule.regions.getRegionMasksCircle()
    
    
    for z,mask in masks.iteritems():
        zi = int(z)
        mask = scipy.ndimage.zoom(mask, ZOOM_FACTOR_3D)
        #pylab.imshow(mask, cmap=pylab.gray())
        #pylab.show()
        row,col = mask.shape
        # take random pixels (every other pixel?)
        
        for i in np.arange(0,row):
            for j in np.arange(0,col):
                if len(pixelTraining) == 0:
                    pixelTraining = [i,j,zi,0]
                else:   
                    if mask[i,j] == False:
                        # make y list: score 1 for nodule-pixels and 0 for nonnodule-pixels
                        pixelTraining = np.vstack([pixelTraining, [i,j,zi,0]])
                    else:
                        pixelTraining = np.vstack([pixelTraining, [i,j,zi,1]])
print(pixelTraining)
a = sum(pixelTraining)
print(a)

        
# store trainingsdata for further use
import pickle
pixelTraining001 = pixelTraining # give specific name to trainingsset
f = open('pixelTraining_LIDC001.pkl', 'wb') # give name to document
pickle.dump(pixelTraining001, f, pickle.HIGHEST_PROTOCOL)
f.close()

######################## FOR LARGE DATASETS
# import tables
# h5file = tables.openFile('test.h5', mode='w', title="Test Array")
# root = h5file.root
# h5file.createArray(root, "test", a)
# h5file.close()     
                 
######################## OPEN pickle
#pixelTraining2 = pickle.load( open( 'pixelTraining_LIDC001.pkl', 'rb') )
#print(np.array_equal(pixelTraining2, pixelTraining001))

    
 