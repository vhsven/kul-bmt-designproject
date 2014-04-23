import numpy as np
import numpy.ma as ma
import pylab as pl
from matplotlib.widgets import Slider
#from sklearn.cross_validation import cross_val_score
from DicomFolderReader import DicomFolderReader
from Trainer import Trainer
from Classifier import Classifier

def update(val):
    _threshold = tSlider.val
    _mySlice = int(sSlider.val)
    _data = c.dfr.getSlicePixelsRescaled(_mySlice)
    _probImg = probImg3D[:,:,_mySlice]
    _masked = ma.masked_greater_equal(_probImg, _threshold)
    _mask = _masked.mask
    
    sp1.clear()
    sp2.clear()
    sp3.clear()
    
    sp1.imshow(_data, cmap=pl.gray())
    sp2.imshow(_probImg, cmap=pl.cm.jet)  # @UndefinedVariable ignore
    sp3.imshow(_mask, cmap=pl.gray())
    
    fig.canvas.draw_idle()
           
paths = int(raw_input("Enter #datasets: "))+1
trainer = Trainer("../data/LIDC-IDRI", maxPaths=paths)

mySlice = 89
myPath = DicomFolderReader.findPath("../data/LIDC-IDRI", 1)
c = Classifier(myPath)

#points2D = Classifier.generatePixelList2D((h, w))
#points3D = Classifier.generatePixelList3D((h, w, d))
#points3D = c.dfr.getThresholdPixels()
mask3D = c.dfr.getThresholdMask()
    
for level in range(1, 3):
    print("Cascade level {}".format(level))
    #Phase 1: training
    clf = trainer.train(level)

    #Phase 2: test model
    c.setLevel(level, clf)
    
    #probImg, mask = c.generateProbabilityImage(mask2D, mySlice, threshold=0.01)
    probImg3D, mask3D = c.generateProbabilityVolume(mask3D, threshold=0.01)
    #probImg = probImg3D[:,:,mySlice]
    #mask = mask3D[:,:,mySlice]
    
    #TODO probImg wegschrijven naar DICOM/tiff
    
    fig, ax = pl.subplots()
    pl.subplots_adjust(bottom=0.20)
     
    sp1 = pl.subplot(131)
    sp2 = pl.subplot(132)
    sp3 = pl.subplot(133)
     
    h,w,d = c.dfr.getVolumeShape()
    #axes: left, bottom, width, height
    sSlider = Slider(pl.axes([0.1, 0.10, 0.8, 0.03]), 'Slice', 0, d-1, 90, valfmt='%1.0f')
    tSlider = Slider(pl.axes([0.1, 0.05, 0.8, 0.03]), 'Threshold', 0.0, 1.0, 0.5)
     
    sSlider.on_changed(update)
    tSlider.on_changed(update)
    update(0)
    pl.show()
    
#TODO download more datasets