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
for nodule in reader.Nodules:
    centerMap = nodule.regions.getRegionCenters()
    for pixelZ,centers in centerMap.iteritems():
        print("Slice z={}".format(pixelZ))
        print(centers)

n_estimators = 30
model = ExtraTreesClassifier(n_estimators=n_estimators)

X=0 #featurevector per datapoint
y=0 #class per datapoint

#clf = clone(model)
#clf = model.fit(X, y)