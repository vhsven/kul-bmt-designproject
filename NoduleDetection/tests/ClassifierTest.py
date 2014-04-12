import scipy
import numpy as np
import pylab as pl
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from featureselection import FeatureSelection
from XmlAnnotationReader import XmlAnnotationReader
from Constants import *

#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
reader = XmlAnnotationReader(myPath)
data = reader.dfr.getVolumeData()
select = FeatureSelection(data, reader.dfr.getVoxelShape())

features = np.array([])

#global 3D features
edges = select.edges()

for nodule in reader.Nodules:
    print(nodule.ID)
    masks, centerMap, r2 = nodule.regions.getRegionMasksCircle()
    #centerMap = nodule.regions.getRegionCenters()
    for z,mask in masks.iteritems():
        zi = int(z)
        mask = scipy.ndimage.zoom(mask, ZOOM_FACTOR_3D)
        print("\tSlice (z={})".format(z))
        data = reader.dfr.getSlicePixelsRescaled(int(z))
        
        #2D slice features
        sliceEntropy = select.image_entropy(z)
        #entropy2 = select.pixelentropy(z)
        blobs = select.blobdetection(z)
        
        xs, ys = np.where(mask)
        for x,y in zip(xs, ys):
            pixelFeatures = ()
            
            #get pixel features from 3D features
            pixelFeatures += (edges[x,y,zi],)
            #get pixel features from slice features
            pixelFeatures += (sliceEntropy,)
            #pixelFeatures += (entropy2[x,y],)
            for blob in blobs:
                pixelFeatures += (blob[x,y],)
            #pixel features
            pixelFeatures += select.trivialFeature(x,y,zi)
            pixelFeatures += (select.forbeniusnorm(x,y,zi),)
            pixelFeatures += select.greyvaluefrequency(x,y,zi)
            pixelFeatures += select.neighbours(x,y,zi)
            
            for windowSize in np.arange(3,32,2):
                pixelFeatures += select.averaging3D(x,y,zi, windowSize)
                pixelFeatures += select.greyvaluecharateristic(x,y,zi, windowSize)
                pixelFeatures += select.windowFeatures(x,y,zi, windowSize)
            
            if len(features) == 0:
                features = np.array(pixelFeatures)
                features = features.reshape(1, len(pixelFeatures)) #else shape = (n,)
            else:
                features = np.vstack([features, np.array(pixelFeatures)])

n_estimators = 30

#model = RandomForestClassifier(n_estimators=n_estimators)
model = ExtraTreesClassifier(n_estimators=n_estimators)

X=features #featurevector per datapoint
y=np.ones(len(X)) #class per datapoint

clf = clone(model)
clf = model.fit(X, y)
scores = clf.score(X, y)
print(scores)

result = model.predict_proba(X[0,:])
#result = model.predict(X[0,:])
print(result)

#TODO cascaded