import datetime
import numpy as np
import pylab as pl
from matplotlib.widgets import Slider
from DicomFolderReader import DicomFolderReader
from Preprocessor import Preprocessor
from Trainer import Trainer
from Classifier import Classifier
from Validator import Validator
from Constants import CASCADE_THRESHOLD, MAX_LEVEL
from XmlAnnotationReader import XmlAnnotationReader

#TODO mention wall nodules more explicitely?

class Main:
    def __init__(self, rootPath, maxPaths=999999, maxLevel=-1):
        self.RootPath = rootPath
        self.MaxPaths = maxPaths
        if maxLevel == -1:
            self.MaxLevel = MAX_LEVEL
        else:
            self.MaxLevel = maxLevel
        
    def main(self):
        print datetime.datetime.now()
        trainer = Trainer(self.RootPath, 0, maxPaths=self.MaxPaths)
        print("Phase 1: training all datasets up to level {}.".format(self.MaxLevel))
        #models = trainer.trainAll(self.MaxLevel, save=False)
        #models = trainer.trainAndValidateAll(self.MaxLevel, save=True)
        models = Trainer.loadAll(self.MaxLevel)
        del trainer
        print("Training phase completed, start testing phase.")
        
        #Phase 2: test models
        totalTP = 0
        totalFP = 0
        totalFN = 0
        nbTestSets = 0
        print datetime.datetime.now()
        for testSet in range(50,51): #DicomFolderReader.findPathsByID(self.RootPath, range(31,51)):
            try:
                dfr = DicomFolderReader.create(self.RootPath, testSet)
                nbTestSets += 1
            except:
                continue
            print("Processing test set {}: '{}'".format(testSet, dfr.Path))
            dfr.printInfo(prefix="\t")
            data = dfr.getVolumeData()
            vshape = dfr.getVoxelShape()
            reader = dfr.getAnnotationReader()
            noduleMask3D = reader.getNodulesMask(data.shape, max, 1.5)
            mask3D = Preprocessor.loadThresholdMask(testSet) #getThresholdMask(data)
            clf = Classifier(testSet, data, vshape)
            
            nbVoxels = mask3D.sum()
            h,w,d = mask3D.shape
            totalVoxels = h*w*d
            ratio = 100.0 * nbVoxels / totalVoxels
            print("\t{0} voxels ({1:.3f}%) remaining after lung segmentation.".format(nbVoxels, ratio))
                
            for level in range(1, self.MaxLevel+1):
                print("\tTesting cascade level {}".format(level))
                clf.setLevel(level, models[level])

                probImg3D, mask3D = clf.generateProbabilityVolume(mask3D, threshold=CASCADE_THRESHOLD[level])

                nbVoxels = mask3D.sum()
                h,w,d = mask3D.shape
                totalVoxels = h*w*d
                ratio = 100.0 * nbVoxels / totalVoxels
                print("\t{0} voxels ({1:.3f}%) remaining after level {2}.".format(nbVoxels, ratio, level))
        
                show = True
                if show:
                    fig, _ = pl.subplots()
                    pl.subplots_adjust(bottom=0.20)
                    sp1 = pl.subplot(221)
                    sp2 = pl.subplot(222)
                    sp3 = pl.subplot(223)
                    sp4 = pl.subplot(224)
                     
                    #axes: left, bottom, width, height
                    sSlider = Slider(pl.axes([0.1, 0.10, 0.8, 0.03]), 'Slice', 0, dfr.getNbSlices()-1, 50, valfmt='%d')
                    tSlider = Slider(pl.axes([0.1, 0.05, 0.8, 0.03]), 'Threshold', 0.0, 1.0, CASCADE_THRESHOLD[level])
                    
                    def update(val):
                        _threshold = tSlider.val
                        _mySlice = int(sSlider.val)
                        _data = dfr.getSliceDataRescaled(_mySlice)
                        _probImg = probImg3D[:,:,_mySlice]
                        _mask = _probImg >= _threshold
                        _nodMask = noduleMask3D[:,:,_mySlice]
                        _nodMasked = np.ma.array(_data, mask=~_nodMask)
                        
                        sp1.clear()
                        sp2.clear()
                        sp3.clear()
                        sp4.clear()
                        
                        sp1.imshow(_data, cmap='bone')
                        sp2.imshow(_nodMasked, cmap='bone')
                        sp3.imshow(_probImg, cmap='jet')
                        sp4.imshow(_mask, cmap='gray')
                        
                        sp1.set_title('Slice {}'.format(_mySlice))
                        sp2.set_title('Nodules')
                        sp3.set_title('Probability Image')
                        sp4.set_title('Thresholded ProbImg')
                        
                        sp1.axis('off')
                        sp2.axis('off')
                        sp3.axis('off')
                        sp4.axis('off')
                        
                        fig.canvas.draw_idle()
                     
                    sSlider.on_changed(update)
                    tSlider.on_changed(update)
                    update(0)
                    pl.show()
                
            print("Finished classifying test set {0}.".format(testSet))
            
            reader = XmlAnnotationReader(dfr.Path, dfr.getCoordinateConverter())
            val = Validator(reader.Nodules)
            nbTP, nbFN, positives = val.searchNodules(mask3D)
            nbFP = val.searchFPs(positives)
            #_, nbTP, nbFP, nbFN = val.ValidateData(clusteredData)
            print "TP={}, FP={}, FN={}".format(nbTP, nbFP, nbFN)
            totalTP += nbTP
            totalFP += nbFP
            totalFN += nbFN
        
        meanTP = totalTP / float(nbTestSets)
        meanFP = totalFP / float(nbTestSets)
        meanFN = totalFN / float(nbTestSets)
        sensitivity = totalTP / float(totalTP + totalFN)
        precision = totalTP / float(totalTP + totalFP)
        print "Means: TP={}, FP={}, FN={}".format(meanTP, meanFP, meanFN)
        print "Sensitivity={}, Precision={}".format(sensitivity, precision)
        print datetime.datetime.now()
    
if __name__ == "__main__":
    maxPaths = int(raw_input("Enter # training datasets: "))
    maxLevel = int(raw_input("Enter max training level: "))
    m = Main("../data/LIDC-IDRI", maxPaths, maxLevel)
    m.main()
