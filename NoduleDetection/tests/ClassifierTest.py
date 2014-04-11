import numpy as np
import pylab as pl
from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from featureselection import FeatureSelection

#from sklearn.externals.six.moves import xrange
from XmlAnnotationReader import XmlAnnotationReader

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
reader = XmlAnnotationReader(myPath)
data = reader.dfr.getVolumeData()
select = FeatureSelection(data)

features = []

#global 3D features
edges = select.edges()

for nodule in reader.Nodules:
    print(nodule.ID)
    masks, centerMap, r2 = nodule.regions.getRegionMasksCircle()
    #centerMap = nodule.regions.getRegionCenters()
    for z,mask in masks.iteritems():
        print("\tSlice (z={})".format(z))
        data = reader.dfr.getSlicePixelsRescaled(int(z))
        
        #2D slice features
        sliceEntropy = select.image_entropy(z)
        entropy2 = select.pixelentropy(z)
        blobs = select.blobdetection(z)
        
        x, y = np.where(mask)
        for pixel in zip(x, y):
            pixelFeatures = () #TODO bereken feature vector van pixel
            
            #get pixel features from 3D features
            pixelFeatures += edges[x,y,z]
            #get pixel features from slice features
            pixelFeatures += sliceEntropy
            pixelFeatures += entropy2[x,y]
            pixelFeatures += blobs[x,y]
            #pixel features
            pixelFeatures += select.trivialFeature(x, y, z)
            pixelFeatures += select.forbeniusnorm(x, y, z)
            pixelFeatures += select.greyvaluefrequency(x, y, z)
            pixelFeatures += select.neighbours(x, y, z)
            
            for windowSize in np.arange(3,51,2):
                pixelFeatures += select.averaging3D(x, y, z, windowSize)
                pixelFeatures += select.greyvaluecharateristic(x, y, z, windowSize)
                pixelFeatures += select.windowFeatures(x, y, z, windowSize)
            
            features += [pixelFeatures]
        

n_estimators = 30
#model = RandomForestClassifier(n_estimators=n_estimators)
model = ExtraTreesClassifier(n_estimators=n_estimators)

X=features #featurevector per datapoint
y=np.ones(len(X)) #class per datapoint

clf = clone(model)
clf = model.fit(X, y)

#TODO cascaded