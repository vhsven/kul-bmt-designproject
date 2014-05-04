import pylab as pl
import numpy as np

#from skimage.transform.pyramids import pyramid_laplacian
from pyramids import pyramid_laplacian
from DicomFolderReader import DicomFolderReader

myPath = DicomFolderReader.findPath("../data/LIDC-IDRI", 1)
dfr = DicomFolderReader(myPath)
data = dfr.getVolumeData()


sigma = 2.5
sigma = np.array([sigma]*3) / dfr.getVoxelShape()
print sigma
for result in pyramid_laplacian(data, sigma=sigma):
    h,w,d = result.shape
    #mySlice = int(90.0 / 133.0 * d)
    img = result[:,:, 90]
    pl.imshow(img, cmap=pl.cm.jet) # @UndefinedVariable
    pl.show()