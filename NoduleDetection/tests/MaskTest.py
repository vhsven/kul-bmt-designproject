import dicom
import pylab as pl
import numpy as np
import numpy.ma as ma
from DicomFolderReader import DicomFolderReader

myPath = DicomFolderReader.findPath("../data/LIDC-IDRI", 1)
dfr = DicomFolderReader(myPath)
data = dfr.getVolumeData()

mask = dicom.read_file("../data/LIDC-Masks/img1.dcm").pixel_array.astype(bool)
mask = np.rollaxis(mask, 0, 3)
masked = ma.array(data=data, mask=~mask)

pl.imshow(masked[:,:,89], cmap=pl.gray())
pl.show()