#import numpy as np
import pylab as pl
from matplotlib.widgets import Slider
from DicomFolderReader import DicomFolderReader
from Preprocessor import Preprocessor
from Trainer import Trainer
from Classifier import Classifier
from Validator import Validator
from Constants import CASCADE_THRESHOLD
from XmlAnnotationReader import XmlAnnotationReader

#remove datasets because only 1 voxel nodules: 2,19,22,25,28,29,32,35,38
#TODO validation + optimale params -> level 3+
#TODO check wall nodules
#TODO calculate nodule radius stats
#Report: better reuse featurevectors from previous level 

class Main:
    def __init__(self, rootPath, testSet, maxPaths=999999):
        self.RootPath = rootPath
        self.MaxPaths = maxPaths
        self.TestSet = testSet
        myPath = DicomFolderReader.findPath(self.RootPath, testSet)
        self.dfr = DicomFolderReader(myPath)
        self.dfr.compress()
    
    def printInfo(self):
        cc = self.dfr.getCoordinateConverter()
        reader = XmlAnnotationReader(self.dfr.Path, cc)
        print("Nodule positions and radii in dataset {}: ".format(self.TestSet))
        for c, r in reader.getNodulePositions():
            print c, r
        print("")
        
    def main(self):
        self.printInfo()
    
        data = self.dfr.getVolumeData()
        vshape = self.dfr.getVoxelShape()
        trainer = Trainer(self.RootPath, self.TestSet, maxPaths=self.MaxPaths)
        clf = Classifier(self.TestSet, data, vshape)
        
        #mask3D = Preprocessor.getThresholdMask(data)
        mask3D = Preprocessor.loadThresholdMask(self.TestSet)
        for level in range(1, 6):
            print("Cascade level {}".format(level))
            #Phase 1: training
            #if level <= 0: #use when previous run failed but saved some models
            #    model = trainer.load(level)
            #else:
            #    model = trainer.trainAndValidate(level)
            #    Trainer.save(model, level)
            
            model = trainer.train(level)
            
            #Phase 2: test model
            clf.setLevel(level, model)

            probImg3D, mask3D = clf.generateProbabilityVolume(mask3D, threshold=CASCADE_THRESHOLD)

            fig, _ = pl.subplots()
            pl.subplots_adjust(bottom=0.20)
             
            sp1 = pl.subplot(131)
            sp2 = pl.subplot(132)
            sp3 = pl.subplot(133)
             
            #axes: left, bottom, width, height
            sSlider = Slider(pl.axes([0.1, 0.10, 0.8, 0.03]), 'Slice', 0, self.dfr.getNbSlices()-1, 50, valfmt='%1.0f')
            tSlider = Slider(pl.axes([0.1, 0.05, 0.8, 0.03]), 'Threshold', 0.0, 1.0, CASCADE_THRESHOLD)
            
            def update(val):
                _threshold = tSlider.val
                _mySlice = int(sSlider.val)
                _data = self.dfr.getSliceDataRescaled(_mySlice)
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
            
            #probImg3D = None #free some memory
        
        h,w,d = mask3D.shape
        nbVoxels = mask3D.sum()
        totalVoxels = h*w*d
        ratio = 100.0 * nbVoxels / totalVoxels
        print("Done, {0} ({1:.2f}%) voxels remaining.".format(nbVoxels, ratio))
        
        #TODO use validator on multiple test sets 
        val = Validator(self.dfr.Path, self.dfr.getCoordinateConverter())
        nodSeg = val.ClusteringData(probImg3D, self.TestSet)
        NodGegT, NodGegF, lijstje, AmountTP, AmountFP, AmountFN = val.ValidateData(nodSeg)
        print('amount of TP')
        print AmountTP
        print('amount of FP')
        print AmountFP
        print('amount of FN')
        print AmountFN
        
testSet = int(raw_input("Enter dataset # to be classified: "))
maxPaths = int(raw_input("Enter # training datasets: "))
m = Main("../data/LIDC-IDRI", testSet, maxPaths)
m.main()
