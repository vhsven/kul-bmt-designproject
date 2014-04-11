import pylab
import numpy as np
import numpy.ma as ma
from XmlAnnotationReader import XmlAnnotationReader

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
reader = XmlAnnotationReader(myPath)
cc = reader.dfr.getCoordinateConverter()
for nodule in reader.Nodules:
    print(nodule.ID)
    #nodule.regions.printRegions()
#     paths, masks = nodule.regions.getRegionMasksPolygon()
#     
#     for pixelZ in masks.keys():
#         mask = masks[pixelZ]
#         mySlice = reader.dfr.getSlicePixelsRescaled(int(pixelZ))
#         mask = np.logical_not(mask)
#         maskedSlice = ma.array(mySlice, mask=mask)
#         pylab.imshow(maskedSlice, cmap=pylab.gray())
#         pylab.show()
    
    mask = nodule.regions.getRegionMasksSphere(cc)
    for pixelZ in range(0, mask.shape[2]):
        mask2D = mask[:,:,pixelZ]
        mySlice = reader.dfr.getSlicePixelsRescaled(int(pixelZ))
        mask2D = np.logical_not(mask2D)
        maskedSlice = ma.array(mySlice, mask=mask2D)
        pylab.imshow(maskedSlice, cmap=pylab.gray())
        pylab.show()
        
        