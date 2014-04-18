import numpy as np
import pylab
from DicomFolderReader import DicomFolderReader

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
dfr = DicomFolderReader(myPath)

for i in np.arange(0, dfr.NbSlices, 10):
    img = dfr.getSlicePixelsRescaled(i)
    img = img // 10
    
    pylab.subplot(121)
    pylab.imshow(img, cmap=pylab.gray())
    
    pylab.subplot(122)
    p, getVolumeEdges = np.histogram(img, 100)
    pylab.bar(getVolumeEdges[:-1], p)
    
    pylab.show()