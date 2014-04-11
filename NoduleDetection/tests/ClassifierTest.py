import numpy as np
import pylab as pl
from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from sklearn.externals.six.moves import xrange
from XmlAnnotationReader import XmlAnnotationReader

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
reader = XmlAnnotationReader(myPath)
features = []
for nodule in reader.Nodules:
    print(nodule.ID)
    masks, centerMap, r2 = nodule.regions.getRegionMasksCircle()
    #centerMap = nodule.regions.getRegionCenters()
    for pixelZ,mask in masks.iteritems():
        print("\tSlice (z={})".format(pixelZ))
        data = reader.dfr.getSlicePixelsRescaled(int(pixelZ))
        x, y = np.where(mask)
        for pixel in zip(x, y):
            pixelFeatures = [] #TODO bereken feature vector van pixel
            features += pixelFeatures
        

n_estimators = 30
#model = RandomForestClassifier(n_estimators=n_estimators)
model = ExtraTreesClassifier(n_estimators=n_estimators)

X=features #featurevector per datapoint
y=1 #class per datapoint

clf = clone(model)
clf = model.fit(X, y)

#TODO cascaded