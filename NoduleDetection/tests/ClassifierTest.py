import scipy
import numpy as np
import pylab as pl
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from featureselection import FeatureSelection
from XmlAnnotationReader import XmlAnnotationReader
from Constants import *

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
reader = XmlAnnotationReader(myPath)
data = reader.dfr.getVolumeData()
select = FeatureSelection(data, reader.dfr.getVoxelShape())


def calculatePixelFeatures(select, x,y,z,c, edges, sliceEntropy, blobs): #TODO use class c
    pixelFeatures = ()
            
    #get pixel features from 3D features
    pixelFeatures += (edges[x,y,z],)
    #get pixel features from slice features
    pixelFeatures += (sliceEntropy,)
    #pixelFeatures += (entropy2[x,y],)
    for blob in blobs:
        pixelFeatures += (blob[x,y],)
    #pixel features
    pixelFeatures += select.trivialFeature(x,y,z)
    pixelFeatures += (select.forbeniusnorm(x,y,z),)
    pixelFeatures += select.greyvaluefrequency(x,y,z)
    pixelFeatures += select.neighbours(x,y,z)
    
    for windowSize in np.arange(3,MAX_FEAT_WINDOW,2):
        pixelFeatures += select.averaging3D(x,y,z, windowSize)
        pixelFeatures += select.greyvaluecharateristic(x,y,z, windowSize)
        pixelFeatures += select.windowFeatures(x,y,z, windowSize)
    
    return pixelFeatures
        
features = np.array([])

#global 3D features
edges = select.edges()

# prevZ = -1;
# for x,y,z,c in classes:
#     if z != prevZ: #2D slice features
#         prevZ = z
#         sliceEntropy = select.image_entropy(z)
#         #entropy2 = select.pixelentropy(z)
#         blobs = select.blobdetection(z)
#     
#     pixelFeatures = calculatePixelFeatures(select, x, y, z, c, edges, sliceEntropy, blobs)
#     if len(features) == 0:
#         features = np.array(pixelFeatures)
#         features = features.reshape(1, len(pixelFeatures)) #else shape = (n,)
#     else:
#         features = np.vstack([features, np.array(pixelFeatures)])
    
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
            pixelFeatures = calculatePixelFeatures(select, x, y, zi, 1, edges, sliceEntropy, blobs)
            
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

#joblib.dump(clf, '../data/models/model.pkl')
#clf = joblib.load('../data/models/model.pkl')

result = clf.predict_proba(X[0,:])
print(result)

#TODO cascaded