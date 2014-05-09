#import numpy as np
import pylab as pl
from matplotlib.widgets import Slider
from DicomFolderReader import DicomFolderReader
from Preprocessor import Preprocessor
from Trainer import Trainer
from Classifier import Classifier
from Validator import Validator
from Constants import CASCADE_THRESHOLD, MAX_LEVEL
from XmlAnnotationReader import XmlAnnotationReader

#remove datasets because only 1 voxel nodules: 2,19,22,25,28,29,32,35,38
#TODO validation + optimale params -> level 4+
#TODO check wall nodules
#Report: better reuse featurevectors from previous level 

class Main:
    def __init__(self, rootPath, maxPaths=999999, maxLevel=-1):
        self.RootPath = rootPath
        self.MaxPaths = maxPaths
        if maxLevel == -1:
            self.MaxLevel = MAX_LEVEL
        else:
            self.MaxLevel = maxLevel
        
    def main(self):    
        #Phase 1: train models
        trainer = Trainer(self.RootPath, 0, maxPaths=self.MaxPaths) #TODO why level per level?
        models = {}
        for level in range(1, self.MaxLevel+1):
            print("Training cascade level {}".format(level))
            #if level <= 0: #use when previous run failed but saved some models
            #    model = trainer.load(level)
            #else:
            #    model = trainer.trainAndValidate(level)
            #    Trainer.save(model, level)
            #model = trainer.loadOrTrain(level)
            model = trainer.train(level)
            
            models[level] = model
        del trainer
        print("Training phase completed, start testing phase...")
    
        #Phase 2: test models
        totalTP = 0
        totalFP = 0
        totalFN = 0
        for testSet in range(31,51): #DicomFolderReader.findPathsByID(self.RootPath, range(31,51)):
            dfr = DicomFolderReader.create(self.RootPath, testSet)
            dfr.printInfo()
            data = dfr.getVolumeData()
            vshape = dfr.getVoxelShape()
            mask3D = Preprocessor.loadThresholdMask(testSet) #getThresholdMask(data)
            clf = Classifier(testSet, data, vshape)
        
            for level in range(1, self.MaxLevel+1):
                print("Test cascade level {}".format(level))
                clf.setLevel(level, models[level])

                probImg3D, mask3D = clf.generateProbabilityVolume(mask3D, threshold=CASCADE_THRESHOLD)

                fig, _ = pl.subplots()
                pl.subplots_adjust(bottom=0.20)
                 
                sp1 = pl.subplot(131)
                sp2 = pl.subplot(132)
                sp3 = pl.subplot(133)
                 
                #axes: left, bottom, width, height
                sSlider = Slider(pl.axes([0.1, 0.10, 0.8, 0.03]), 'Slice', 0, dfr.getNbSlices()-1, 50, valfmt='%1.0f')
                tSlider = Slider(pl.axes([0.1, 0.05, 0.8, 0.03]), 'Threshold', 0.0, 1.0, CASCADE_THRESHOLD)
                
                def update(val):
                    _threshold = tSlider.val
                    _mySlice = int(sSlider.val)
                    _data = dfr.getSliceDataRescaled(_mySlice)
                    _probImg = probImg3D[:,:,_mySlice]
                    _mask = _probImg >= _threshold
                    
                    sp1.clear()
                    sp2.clear()
                    sp3.clear()
                    
                    sp1.imshow(_data, cmap=pl.gray())
                    sp2.imshow(_probImg, cmap=pl.cm.jet)  # @UndefinedVariable ignore
                    sp3.imshow(_mask, cmap=pl.gray())
                    
                    fig.canvas.draw_idle()
                 
                sSlider.on_changed(update)
                tSlider.on_changed(update)
                update(0)
                pl.show()
                
            h,w,d = mask3D.shape
            nbVoxels = mask3D.sum()
            totalVoxels = h*w*d
            ratio = 100.0 * nbVoxels / totalVoxels
            print("Done processing test set {0}, {1} ({2:.2f}%) voxels remaining.".format(testSet, nbVoxels, ratio))
            
            val = Validator(dfr.Path, dfr.getCoordinateConverter())
            nodSeg = val.ClusteringData(probImg3D, testSet)
            NodGegT, NodGegF, lijstje, nbTP, nbFP, nbFN = val.ValidateData(nodSeg)
            print "TP: {}, FP: {}, FN: {}".format(nbTP, nbFP, nbFN)
            totalTP += nbTP
            totalFP += nbFP
            totalFN += nbFN
            
        print "Totals: TP: {}, FP: {}, FN: {}".format(totalTP, totalFP, totalFN)
        
        #TODO totalTN
        totalTN = 0
        sensitivity = totalTP / float(totalTP + totalFN)
        specificity = totalTN / float(totalTN + totalFN)
        
maxPaths = int(raw_input("Enter # training datasets: "))
maxLevel = int(raw_input("Enter max training level: "))
m = Main("../data/LIDC-IDRI", maxPaths, maxLevel)
m.main()
