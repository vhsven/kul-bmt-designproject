import numpy as np
import pylab as pl
#from sklearn.cross_validation import cross_val_score
from DicomFolderReader import DicomFolderReader
from Trainer import Trainer
from Classifier import Classifier

trainer = Trainer("../data/LIDC-IDRI", maxPaths=9999)
clf = trainer.train()

#Test model
mySlice = 10
myPath = DicomFolderReader.findPath("../data/LIDC-IDRI", 1)

c = Classifier(myPath, clf, level=1)
h,w,d = c.dfr.getVolumeShape()
sData = c.dfr.getSlicePixelsRescaled(mySlice)

h //= 2
w //= 2
d //= 8
print h,w,d

#points2D = Classifier.generatePixelList2D((h, w))
#probImg, masked = c.generateProbabilityImage((h,w), points2D, mySlice)

points3D = Classifier.generatePixelList3D((h, w, d)) #TODO use points from segmentation instead
probImg, masked = c.generateProbabilityVolume((h,w,d), points3D)
probImg = probImg[:,:,mySlice]
mask = masked.mask
masked = mask[:,:,mySlice]

pl.subplot(221)
pl.imshow(sData, cmap=pl.gray())
pl.subplot(222)
pl.imshow(probImg, cmap=pl.cm.jet)  # @UndefinedVariable ignore
pl.subplot(223)
pl.imshow(masked, cmap=pl.gray())
pl.show()

points3D = points3D #get points from mask and launch second classifier
#train level 2
#classify level 2


#TODO cascaded: 
#TODO download more datasets
