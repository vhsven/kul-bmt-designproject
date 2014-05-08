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
        print("\tFound {} nodule(s).".format(nbNodules))
        assert nbNodules > 0
        
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
        
        print("Training and validating level {} classifier...".format(level))
        rf = RandomForestClassifier(n_estimators=30) #, n_jobs=-1 
        #cross_validation.KFold(len(x), n_folds=10, indices=True, shuffle=True, random_state=4)
        #X_train, X_test, y_train, y_test = cross_validation.train_test_split(allFeatures, allClasses, test_size=0.5, random_state=0)
        tuned_parameters = [{'min_samples_leaf': np.arange(5, 200, 10)}] #, 'min_samples_split': np.arange(5, 100, 10)
        rfGrid = GridSearchCV(rf, tuned_parameters, cv=NB_VALIDATION_FOLDS)        
        rfGrid.fit(allFeatures, allClasses)
        model = rfGrid.best_estimator_
        #print(rfGrid.best_estimator_)
        print(rfGrid.best_score_)
        print(rfGrid.best_params_)
        
#        Level 1: (4 trainingsets)
#           Accuracy: 0.801369484787
#           Params: {'min_samples_split': 85/95, 'min_samples_leaf': 25/95}

#        Level 2: min_samples_leaf=1, min_samples_split=21, n_estimators=31
#           Accuracy: 0.991339491917
#           Params: {'min_samples_split': 35, 'min_samples_leaf': 5} 
        
        return model
    
    @staticmethod
    def pruneFeatures(allFeatures, allClasses, oldMask, newMask):
        """Selects current level feature out of previous level features based on masks."""
        oldIndices = np.where(oldMask.ravel())[0]
        newIndices = np.where(newMask.ravel())[0]
        indices = np.searchsorted(oldIndices, newIndices)
        return allFeatures[indices,:], allClasses[indices,:]
    
    def train(self, level):
        allFeatures, allClasses = self.calculateAllTrainingFeatures(level)
        
        print("Training level {} classifier...".format(level))
        #TODO set params
        #input, trainset=30 (1-40), testset=50
        
        n_estimators = 30
        min_samples_leaf = 1
        if level == 1: #accuracy = 0.802394890899
            min_samples_leaf = 85
        elif level == 2: #accuracy = 0.980574773816
            min_samples_leaf = 5
        elif level == 3: #accuracy = 0.982597126131
            min_samples_leaf = 5
        elif level == 4: #accuracy = 
            min_samples_leaf = 1
        elif level == 5: #accuracy = 
            min_samples_leaf = 1
            
        rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
        model = rf.fit(allFeatures, allClasses)
        
        return model
    
    @staticmethod
    def save(model, level):
        myFile = "../data/models/model_{}.pkl".format(level)
        print("\tSaved level {} classifier".format(level))
        joblib.dump(model, myFile)
        
    @staticmethod
    def load(level):
        myFile = "../data/models/model_{}.pkl".format(level)
        model = joblib.load(myFile)
        print("\tLoaded level {} classifier".format(level))
        return model
    
    def loadOrTrain(self, level):
        try:
            model = Trainer.load(level)
            return model
        except:
            return self.train(level)