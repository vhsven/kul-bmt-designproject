import numpy as np
from collections import deque
from FeatureGenerator import FeatureGenerator
from XmlAnnotationReader import XmlAnnotationReader
from PixelFinder import PixelFinder
from DicomFolderReader import DicomFolderReader
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier  # @UnusedImport
from sklearn.externals import joblib

class Trainer:
    def __init__(self, rootPath, maxPaths=99999, level=1):
        self.RootPath = rootPath
        self.MaxPaths = maxPaths
        
    def calculateSetTrainingFeatures(self, myPath, level):
        print("Processing '{}'".format(myPath))
        reader = XmlAnnotationReader(myPath)
        print("\tFound {} nodules.".format(len(reader.Nodules)))
        data = reader.dfr.getVolumeData()
        fgen = FeatureGenerator(data, reader.dfr.getVoxelShape(), level)
        finder = PixelFinder(reader)
        
        setFeatures = deque()
        
        #Calculate features of nodule pixels 
        nbNodulePixels = 0
        for x,y,z in finder.findNodulePixels(radiusFactor=0.33):
            nbNodulePixels += 1
            pixelFeatures = fgen.calculatePixelFeatures(x, y, z)
            setFeatures.append(pixelFeatures)
        print("\tFound {} nodules pixels.".format(nbNodulePixels))
        
        #Calculate allFeatures of random non -nodule pixels
        for x,y,z in finder.findRandomNonNodulePixels(nbNodulePixels):
            pixelFeatures = fgen.calculatePixelFeatures(x, y, z)
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
    
    def calculateAllTrainingFeatures(self, level):
        allFeatures = None
        allClasses = None
        for myPath in DicomFolderReader.findPaths(self.RootPath, self.MaxPaths):
            if "LIDC-IDRI-0001" in myPath:
                continue        
            setFeatures, setClasses = self.calculateSetTrainingFeatures(myPath, level)
            if allFeatures is None:
                allFeatures = setFeatures
                allClasses = setClasses
            else:
                allFeatures = np.concatenate([allFeatures, setFeatures], axis=0)
                allClasses = np.concatenate([allClasses, setClasses], axis=0)
        
        #print allFeatures
        #print allClasses
        
        return allFeatures, allClasses
    
    def train(self, level):
        allFeatures, allClasses = self.calculateAllTrainingFeatures(level)
        
        #model = RandomForestClassifier(n_estimators=30)
        model = ExtraTreesClassifier() #n_estimators=30
        clf = clone(model)
        clf = model.fit(allFeatures, allClasses)
        scores = clf.score(allFeatures, allClasses)
        #scores2 = cross_val_score(clf, allFeatures, classes)
        print("Score: {}".format(scores))
        
        return clf
    
    def trainAndSave(self, level, myFile='../data/models/model.pkl'):
        clf = self.train(level)
        joblib.dump(clf, myFile)
        
    @staticmethod
    def loadTraining(myFile='../data/models/model.pkl'):
        return joblib.load(myFile)