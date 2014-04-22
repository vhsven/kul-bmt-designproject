import numpy as np
import numpy.ma as ma
import pylab as pl
from matplotlib.widgets import Slider
#from sklearn.cross_validation import cross_val_score
from DicomFolderReader import DicomFolderReader
from Trainer import Trainer
from Classifier import Classifier

paths = int(raw_input("Enter #datasets: "))+1
trainer = Trainer("../data/LIDC-IDRI", maxPaths=paths)
clf = trainer.train(level=1)

#Test model
mySlice = 89
myPath = DicomFolderReader.findPath("../data/LIDC-IDRI", 1)

#from XmlAnnotationReader import XmlAnnotationReader
#from PixelFinder import PixelFinder
#reader = XmlAnnotationReader(myPath)
#finder = PixelFinder(reader)
#finder.plotHistograms()

c = Classifier(myPath, clf, level=1)
h,w,d = c.dfr.getVolumeShape()
sData = c.dfr.getSlicePixelsRescaled(mySlice)

#h //= 2
#w //= 2
#d //= 8
print h,w,d

points2D = Classifier.generatePixelList2D((h, w))
probImg, masked = c.generateProbabilityImage((h,w), points2D, mySlice)
#points3D = Classifier.generatePixelList3D((h, w, d))
#points3D = c.dfr.getThresholdPixels()
#probImg, masked = c.generateProbabilityVolume((h,w,d), points3D)
#probImg = probImg[:,:,mySlice]
#mask = masked.mask
#masked = mask[:,:,mySlice]

fig, ax = pl.subplots()
pl.subplots_adjust(bottom=0.20)

sp1 = pl.subplot(131)
sp2 = pl.subplot(132)
sp3 = pl.subplot(133)

#axes: left, bottom, width, height
tSlider = Slider(pl.axes([0.1, 0.05, 0.8, 0.03]), 'Threshold', 0.0, 1.0, 0.5)

sp1.imshow(sData, cmap=pl.gray())
sp2.imshow(probImg, cmap=pl.cm.jet)  # @UndefinedVariable ignore

def update(threshold):
    masked = ma.masked_greater_equal(probImg, threshold)
    mask = masked.mask
    sp3.clear()    
    sp3.imshow(mask, cmap=pl.gray())
    fig.canvas.draw_idle()
    
tSlider.on_changed(update)
update(0)
pl.show()

#TODO:
#clf = trainer.train(2)
#c.setLevel(2)
#download more datasets
