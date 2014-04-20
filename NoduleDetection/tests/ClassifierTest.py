import numpy as np
import pylab as pl
from collections import deque
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from FeatureGenerator import FeatureGenerator
from XmlAnnotationReader import XmlAnnotationReader
from DicomFolderReader import DicomFolderReader
from PixelFinder import PixelFinder
from dicom._dicom_dict import DicomDictionary

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
    
    return probImg

def calculateSetTrainingFeatures(myPath):
    print("Processing '{}'...".format(myPath))
    reader = XmlAnnotationReader(myPath)
    data = reader.dfr.getVolumeData()
    fgen = FeatureGenerator(data, reader.dfr.getVoxelShape())
    finder = PixelFinder(reader)
    
    print reader
    print reader.dfr
    #pl.imshow(data[:,:,89])
    #pl.show()

    setFeatures = deque()
    
    #Calculate features of nodule pixels 
    nbNodulePixels = 0
    for x,y,z in finder.findNodulePixels(radiusFactor=0.33):
        nbNodulePixels += 1
        pixelFeatures = calculatePixelFeatures(fgen, x, y, z)
        setFeatures.append(pixelFeatures)
        
    #Calculate allFeatures of random non -nodule pixels
    for x,y,z in finder.findRandomNonNodulePixels(nbNodulePixels):
        pixelFeatures = calculatePixelFeatures(fgen, x, y, z)
        setFeatures.append(pixelFeatures)
    
    setFeatures = np.array(setFeatures)
    
    #Create classification vector
    setClasses = np.zeros(setFeatures.shape[0], dtype=np.bool)
    setClasses[0:nbNodulePixels] = True
    
    #Let's try not to use too much memory
    del finder
    del fgen
    del data
    del reader
    
    return setFeatures, setClasses

def calculateAllTrainingFeatures(rootPath, maxPaths=99999):
    allFeatures = None
    allClasses = None
    for myPath in DicomFolderReader.findPaths(rootPath, maxPaths):        
        setFeatures, setClasses = calculateSetTrainingFeatures(myPath)
        if allFeatures is None:
            allFeatures = setFeatures
            allClasses = setClasses
        else:
            allFeatures = np.concatenate([allFeatures, setFeatures], axis=0)
            allClasses = np.concatenate([allClasses, setClasses], axis=0)
    
    #print allFeatures
    #print allClasses
    
    return allFeatures, allClasses

allFeatures, allClasses = calculateAllTrainingFeatures("../data/LIDC-IDRI", maxPaths=2)

#model = RandomForestClassifier(n_estimators=30)
model = ExtraTreesClassifier(n_estimators=30)
clf = clone(model)
clf = model.fit(allFeatures, allClasses)
scores = clf.score(allFeatures, allClasses)
#scores2 = cross_val_score(clf, allFeatures, classes)
print("Score: {}".format(scores))

#joblib.dump(clf, '../data/models/model.pkl')
#clf = joblib.load('../data/models/model.pkl')


#Test model
myPath = DicomFolderReader.findPath("../data/LIDC-IDRI", 0)
reader = XmlAnnotationReader(myPath)
vData = reader.dfr.getVolumeData()
sData = reader.dfr.getSlicePixelsRescaled(89)
fgen = FeatureGenerator(vData, reader.dfr.getVoxelShape())

print reader
print reader.dfr

probImg = generateProbabilityImage(reader.dfr, fgen, clf, 89)

pl.subplot(121)
pl.imshow(sData, cmap=pl.gray())
pl.subplot(122)
pl.imshow(probImg, cmap=pl.cm.jet)  # @UndefinedVariable ignore
pl.show()

#TODO cascaded
