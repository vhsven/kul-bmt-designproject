#import numpy as np
import pylab as pl
from matplotlib.widgets import Slider
#from sklearn.cross_validation import cross_val_score
from DicomFolderReader import DicomFolderReader
from Preprocessor import Preprocessor
from Trainer import Trainer
from Classifier import Classifier
        
class Main:
    def __init__(self, rootPath="../data/LIDC-IDRI", testSet=1):
        self.RootPath = rootPath
        myPath = DicomFolderReader.findPath(self.RootPath, testSet)
        self.dfr = DicomFolderReader(myPath)
    
    def main(self):
        paths = int(raw_input("Enter #datasets: "))+1
        trainer = Trainer(self.RootPath, maxPaths=paths)
        
        data = self.dfr.getVolumeData()
        vshape = self.dfr.getVoxelShape()
        c = Classifier(data, vshape)
        
        #mask3D = Preprocessor.getThresholdMask(data)
        mask3D = Preprocessor.loadThresholdMask(1)
        for level in range(1, 3):
            print("Cascade level {}".format(level))
            #Phase 1: training
            clf = trainer.train(level)
            #Trainer.save(clf, file="../data/models/model_{}.pkl".format(level))
            #clf = Trainer.load(file="../data/models/model_{}.pkl".format(level))
            
            #Phase 2: test model
            c.setLevel(level, clf)
            
            probImg3D, mask3D = c.generateProbabilityVolume(mask3D, threshold=0.01)
            
            fig, _ = pl.subplots()
            pl.subplots_adjust(bottom=0.20)
             
            sp1 = pl.subplot(131)
            sp2 = pl.subplot(132)
            sp3 = pl.subplot(133)
             
            #axes: left, bottom, width, height
            sSlider = Slider(pl.axes([0.1, 0.10, 0.8, 0.03]), 'Slice', 0, self.dfr.getNbSlices()-1, 0, valfmt='%1.0f')
            tSlider = Slider(pl.axes([0.1, 0.05, 0.8, 0.03]), 'Threshold', 0.0, 1.0, 0.5)
            
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
        
        h,w,d = mask3D.shape
        nbVoxels = mask3D.sum()
        totalVoxels = h*w*d
        ratio = 100.0 * nbVoxels / totalVoxels
        print("Done, {0} ({1:.2f}%) voxels remaining.".format(nbVoxels, ratio))
        
m = Main()
m.main()