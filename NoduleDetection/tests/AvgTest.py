import pylab as pl
import numpy as np
from DicomFolderReader import DicomFolderReader
from FeatureGenerator import FeatureGenerator
dfr = DicomFolderReader.create("../data/LIDC-IDRI", 50)
data = dfr.getVolumeData()
h,w,d = data.shape
mySlice = 93
mask = np.zeros_like(data, dtype=np.bool)
mask[:,:,mySlice] = 1
vshape = dfr.getVoxelShape()

fgen = FeatureGenerator(50, data, vshape, 1)
result = fgen.averaging3DByMask(mask, windowSize=3, vesselSize=7.5)

result = result.reshape((512,512))

pl.subplot(121)
pl.imshow(data[:,:,mySlice], cmap=pl.cm.bone)  # @UndefinedVariable
pl.subplot(122)
pl.imshow(result, cmap=pl.cm.bone)  # @UndefinedVariable
pl.show()

#TODO avg3D: right effect?