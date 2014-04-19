import numpy as np
import pylab as pl
from collections import deque
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from FeatureGenerator import FeatureGenerator
from XmlAnnotationReader import XmlAnnotationReader
from PixelFinder import PixelFinder

def calculatePixelFeatures(fgen, x,y,z):
    z = int(z)
    
    #global 3D features
    #getVolumeEdges = fgen.getVolumeEdges()
    
    #2D slice features (TODO only calculate once)
    #sliceEntropy = fgen.image_entropy(z)
    #entropy2 = fgen.pixelentropy(z)
    #blobs = fgen.blobdetection(z)
    
    pixelFeatures = ()
            
    #get pixel features from 3D features
    #pixelFeatures += (getVolumeEdges[x,y,z],)
    #get pixel features from slice features
    #pixelFeatures += (sliceEntropy,)
    #pixelFeatures += (entropy2[x,y],)
    #for blob in blobs:
    #    pixelFeatures += (blob[x,y],)
    #pixel features
    pixelFeatures += fgen.getTrivialFeatures(x,y,z)
    #pixelFeatures += (fgen.forbeniusnorm(x,y,z),)
    #pixelFeatures += fgen.neighbours(x,y,z)
    
    #for windowSize in np.arange(3,MAX_FEAT_WINDOW,2):
    #    pixelFeatures += fgen.greyvaluefrequency(x,y,z, windowSize)
    #    pixelFeatures += fgen.averaging3D(x,y,z, windowSize)
    #    pixelFeatures += fgen.greyvaluecharateristic(x,y,z, windowSize)
    #    pixelFeatures += fgen.windowFeatures(x,y,z, windowSize)
    
    return pixelFeatures

def generateProbabilityImage(dfr, fgen, clf, mySlice):
    h, w, _ = dfr.getVolumeShape()
    
    x, y = np.meshgrid(np.arange(h), np.arange(w))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    testFeatures = deque()
    for px,py in points:
        pixelFeatures = calculatePixelFeatures(fgen, px, py, mySlice)
        testFeatures.append(pixelFeatures)

    testFeatures = np.array(testFeatures)
    result = clf.predict_proba(testFeatures)
    probImg = result[:,1]
    probImg = probImg.reshape(h, w).T

    pl.subplot(121)
    pl.imshow(dfr.getSlicePixelsRescaled(mySlice), cmap=pl.gray())
    pl.subplot(122)
    pl.imshow(probImg, cmap=pl.cm.jet)
    pl.show()
    
    return probImg

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
reader = XmlAnnotationReader(myPath)
data = reader.dfr.getVolumeData()
fgen = FeatureGenerator(data, reader.dfr.getVoxelShape())
finder = PixelFinder(reader)

allFeatures = deque()

#Calculate features of nodule pixels 
nbNodulePixels = 0
for x,y,z in finder.findNodulePixels(radiusFactor=0.33):
    nbNodulePixels += 1
    pixelFeatures = calculatePixelFeatures(fgen, x, y, z)
    allFeatures.append(pixelFeatures)
    
#Calculate allFeatures of random non -nodule pixels
for x,y,z in finder.findRandomNonNodulePixels(nbNodulePixels):
    pixelFeatures = calculatePixelFeatures(fgen, x, y, z)
    allFeatures.append(pixelFeatures)

allFeatures = np.array(allFeatures)

#Create classification vector
classes = np.zeros(allFeatures.shape[0], dtype=np.bool)
classes[0:nbNodulePixels] = True

#model = RandomForestClassifier(n_estimators=30)
model = ExtraTreesClassifier(n_estimators=30)
clf = clone(model)
clf = model.fit(allFeatures, classes)
scores = clf.score(allFeatures, classes)
#scores2 = cross_val_score(clf, allFeatures, classes)
print("Score: {}".format(scores))

#joblib.dump(clf, '../data/models/model.pkl')
#clf = joblib.load('../data/models/model.pkl')


#Test model on different dataset
myPath2 = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
reader2 = XmlAnnotationReader(myPath2)
data2 = reader2.dfr.getVolumeData()
fgen2 = FeatureGenerator(data2, reader2.dfr.getVoxelShape())

# pl.imshow(data[:,:,89])
# pl.show()
# pl.imshow(data2[:,:,89])
# pl.show()

probImg = generateProbabilityImage(reader2.dfr, fgen2, clf, 89)

#TODO cascaded
