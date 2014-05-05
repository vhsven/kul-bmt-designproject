import numpy as np
from FeatureGenerator import FeatureGenerator
from PixelFinder import PixelFinder
from DicomFolderReader import DicomFolderReader
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.metrics import classification_report
from sklearn.cross_validation import StratifiedKFold
from Constants import NB_VALIDATION_FOLDS

class Trainer:
    def __init__(self, rootPath, setID, maxPaths=99999):
        self.RootPath = rootPath
        self.MaxPaths = maxPaths
        self.IgnorePath = "LIDC-IDRI-{0:0>4d}".format(setID)
            
    def calculateSetTrainingFeatures(self, myPath, level):
        dfr = DicomFolderReader(myPath)
        dfr.compress()
        setID = dfr.getSetID()
        print("Processing set {}: '{}'".format(setID, myPath))
        cc = dfr.getCoordinateConverter()
        finder = PixelFinder(myPath, cc)
        data = dfr.getVolumeData()
        shape = dfr.getVolumeShape()
        vshape = dfr.getVoxelShape()
        fgen = FeatureGenerator(setID, data, vshape, level)
        nbNodules = len(finder.Reader.Nodules)
        print("\tFound {} nodules.".format(nbNodules))
        
        #pixelsP, pixelsN = finder.getLists(shape, radiusFactor=0.33)
        print("\tProcessing pixels...")
        
        #Calculate features of nodule pixels
        #nbNodulePixels = len(pixelsP)
        #x,y,z = pixelsP[0]
        #setFeatures = fgen.getAllFeatures(x,y,z)
        #for x,y,z in pixelsP[1:]:
        #    pixelFeatures = fgen.getAllFeatures(x, y, z) #1xL ndarray
        #    setFeatures = np.vstack([setFeatures, pixelFeatures])
        #print("\tProcessed {} nodules pixels.".format(nbNodulePixels))
        
        #Calculate allFeatures of random non -nodule pixels
        #for x,y,z in pixelsN:
        #    pixelFeatures = fgen.getAllFeatures(x, y, z)
        #    setFeatures = np.vstack([setFeatures, pixelFeatures])
        #print("\tProcessed {} random non-nodules pixels.".format(nbNodulePixels))
        
        maskP, maskN, nbNodulePixels = finder.getMasks(shape, radiusFactor=0.33)
        featuresP = fgen.getAllFeaturesByMask(maskP)
        print("\tProcessed {} nodules pixels.".format(nbNodulePixels))
        featuresN = fgen.getAllFeaturesByMask(maskN)
        print("\tProcessed {} random non-nodules pixels.".format(nbNodulePixels))
        setFeatures = np.vstack([featuresP, featuresN])
        
        #Create classification vector
        setClasses = np.zeros(setFeatures.shape[0], dtype=np.bool)
        setClasses[0:nbNodulePixels] = True
        
        #Let's try not to use too much memory
        del finder
        del fgen
        del data
        del cc
        del dfr
        
        return setFeatures, setClasses
    
    def calculateAllTrainingFeatures(self, level):
        allFeatures = None
        allClasses = None
        for myPath in DicomFolderReader.findPaths(self.RootPath, self.MaxPaths):
            if self.IgnorePath in myPath:
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
    
    def trainAndValidate(self, level):
        allFeatures, allClasses = self.calculateAllTrainingFeatures(level)
        
        print("Training level {} classifier...".format(level))
        model = RandomForestClassifier(n_estimators=30) #, n_jobs=-1
        print model.get_params()        
        #cross_validation.KFold(len(x), n_folds=10, indices=True, shuffle=True, random_state=4)
        #X_train, X_test, y_train, y_test = cross_validation.train_test_split(allFeatures, allClasses, test_size=0.5, random_state=0)
        tuned_parameters = [{'min_samples_leaf': np.arange(5, 100, 10),  
                             'min_samples_split': np.arange(5, 100, 10)}]
        clfs = GridSearchCV(model, tuned_parameters, cv=NB_VALIDATION_FOLDS)        
        clfs.fit(allFeatures, allClasses)
        clf = clfs.best_estimator_
        #print(clfs.best_estimator_)
        print(clfs.best_score_)
        print(clfs.best_params_)
        
#        Default: min_samples_leaf=1, min_samples_split=2, n_estimators=10

#        Level 1: (4 trainingsets)
#           Accuracy: 0.833718244804
#           Params: {'min_samples_split': 85, 'min_samples_leaf': 25}

#        Level 2: min_samples_leaf=1, min_samples_split=21, n_estimators=31
#           Accuracy: 0.991339491917
#           Params: {'min_samples_split': 35, 'min_samples_leaf': 5} 
        
        #print("Scores: {}".format(scores))
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
        return clf
        
    def train(self, level):
        allFeatures, allClasses = self.calculateAllTrainingFeatures(level)
        
        print("Training level {} classifier...".format(level))
        model = RandomForestClassifier(n_estimators=30, n_jobs=-1)
        clf = model.fit(allFeatures, allClasses)
        
        return clf
    
    @staticmethod
    def save(clf, level):
        myFile = "../data/models/model_{}.pkl".format(level)
        joblib.dump(clf, myFile)
        
    @staticmethod
    def load(level):
        myFile = "../data/models/model_{}.pkl".format(level)
        return joblib.load(myFile)
    
    def loadOrTrain(self, level):
        try:
            clf = Trainer.load(level)
            print("Loaded level {} classifier".format(level))
            return clf
        except:
            return self.train(level)