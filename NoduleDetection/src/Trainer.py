import numpy as np
from FeatureGenerator import FeatureGenerator
from PixelFinder import PixelFinder
from DicomFolderReader import DicomFolderReader
#from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
#from sklearn.metrics.metrics import classification_report
#from sklearn.cross_validation import StratifiedKFold
from Constants import NB_VALIDATION_FOLDS, RADIUS_FACTOR

class Trainer:
    def __init__(self, rootPath, setID, maxPaths=99999):
        self.RootPath = rootPath
        self.MaxPaths = maxPaths
        self.IgnorePath = "LIDC-IDRI-{0:0>4d}".format(setID)
            
    def calculateSetTrainingFeatures(self, myPath, level):
        dfr = DicomFolderReader(myPath)
        dfr.compress()
        setID = dfr.getSetID()
        print("Processing training set {}: '{}'".format(setID, myPath))
        cc = dfr.getCoordinateConverter()
        finder = PixelFinder(myPath, cc)
        data = dfr.getVolumeData()
        shape = dfr.getVolumeShape()
        vshape = dfr.getVoxelShape()
        fgen = FeatureGenerator(setID, data, vshape, level)
        nbNodules = len(finder.Reader.Nodules)
        print("\tFound {} nodule(s).".format(nbNodules))
        assert nbNodules > 0
        
        maskP, maskN, nbNodulePixels = finder.getMasks(shape, radiusFactor=RADIUS_FACTOR)
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
        for myPath in DicomFolderReader.findAllPaths(self.RootPath, self.MaxPaths):
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
    
    def trainAndValidateAll(self, maxLevel, save=False):
        print("Training and validating classifier up to level {}.".format(maxLevel))
        allFeatures, allClasses = self.calculateAllTrainingFeatures(maxLevel)
        
        models = {}
        levelSlices = [0, 1, 1+9, 1+9+1, 1+9+1+1, 1+9+1+1+6]
        for level in range(1, maxLevel+1):
            levelSlice = levelSlices[level]
            features = allFeatures[:, 0:levelSlice]
            rf = RandomForestClassifier(n_estimators=30)
            tuned_parameters = [{'min_samples_leaf': np.arange(5, 100, 10)}]
            rfGrid = GridSearchCV(rf, tuned_parameters, cv=NB_VALIDATION_FOLDS)        
            rfGrid.fit(features, allClasses)
            models[level] = rfGrid.best_estimator_
            print(rfGrid.best_score_)
            print(rfGrid.best_params_)
            
        if save:
            Trainer.saveAll(models)
            
        return models
    
#     def trainAndValidate(self, level):
#         allFeatures, allClasses = self.calculateAllTrainingFeatures(level)
#         
#         print("Training and validating level {} classifier...".format(level))
#         rf = RandomForestClassifier(n_estimators=30) #, n_jobs=-1 
#         #cross_validation.KFold(len(x), n_folds=10, indices=True, shuffle=True, random_state=4)
#         #X_train, X_test, y_train, y_test = cross_validation.train_test_split(allFeatures, allClasses, test_size=0.5, random_state=0)
#         tuned_parameters = [{'min_samples_leaf': np.arange(5, 200, 10)}]
#         rfGrid = GridSearchCV(rf, tuned_parameters, cv=NB_VALIDATION_FOLDS)        
#         rfGrid.fit(allFeatures, allClasses)
#         model = rfGrid.best_estimator_
#         print(rfGrid.best_score_)
#         print(rfGrid.best_params_)
#         
#         return model
    
    def trainAll(self, maxLevel, save=False):
        allFeatures, allClasses = self.calculateAllTrainingFeatures(maxLevel)
        
        #TODO set params
        n_estimators = 30
        
        models = {}
        levelSlices = [0, 1, 1+9, 1+9+1, 1+9+1+1, 1+9+1+1+6]
        for level in range(1, maxLevel+1):
            if level == 1:
                min_samples_leaf = 55
            else:
                min_samples_leaf = 5
            levelSlice = levelSlices[level]
            features = allFeatures[:, 0:levelSlice]
            rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
            models[level] = rf.fit(features, allClasses)
            
        if save:
            Trainer.saveAll(models)
        
        return models
    
#     def train(self, level):
#         allFeatures, allClasses = self.calculateAllTrainingFeatures(level)
#         
#         print("Training level {} classifier...".format(level))
#         #TODO set params
#         #input, trainset=30 (1-40), testset=50
#         
#         n_estimators = 30
#         min_samples_leaf = 1
#         if level == 1: #accuracy = 
#             min_samples_leaf = 55
#         elif level == 2: #accuracy = 
#             min_samples_leaf = 5
#         elif level == 3: #accuracy = 
#             min_samples_leaf = 5
#         elif level == 4: #accuracy = 
#             min_samples_leaf = 1
#         #elif level == 5: #accuracy = 
#         #    min_samples_leaf = 1
#             
#         rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
#         model = rf.fit(allFeatures, allClasses)
#         
#         return model
    
    @staticmethod
    def save(model, level):
        myFile = "../data/models/model_{}.pkl".format(level)
        print("\tSaved level {} classifier".format(level))
        joblib.dump(model, myFile, 3)
        
    @staticmethod
    def saveAll(models):
        for level in models:
            myFile = "../data/models/model_{}.pkl".format(level)
            joblib.dump(models[level], myFile, 3)
        print("\tSaved models up to level {}.".format(level))
        
    @staticmethod
    def load(level):
        myFile = "../data/models/model_{}.pkl".format(level)
        model = joblib.load(myFile)
        print("\tLoaded level {} model".format(level))
        return model
    
    @staticmethod
    def loadAll(maxLevel):
        models = {}
        for level in range(1, maxLevel+1):
            myFile = "../data/models/model_{}.pkl".format(level)
            models[level] = joblib.load(myFile)
        
        print("\tLoaded models up to level {}.".format(maxLevel))
        return models
    
    def loadOrTrain(self, level):
        try:
            model = Trainer.load(level)
            return model
        except:
            return self.train(level)