import pylab
import numpy as np
import numpy.ma as ma
from XmlAnnotationReader import XmlAnnotationReader

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
reader = XmlAnnotationReader(myPath)
cc = reader.dfr.getCoordinateConverter()
m,n,_ = reader.dfr.getVolumeShape()

# for c,r2 in reader.getNodulePositions():
#     print(c,r2)

for nodule in reader.Nodules:
    print(nodule.ID)
    #nodule.regions.printRegions()
    masks, c, r = nodule.regions.getRegionMasksCircle(m,n,0.5)
    #paths, masks = nodule.regions.getRegionMasksPolygon(m,n)
       
    for z in masks.keys():
        mask = masks[z]
        mySlice = reader.dfr.getSlicePixelsRescaled(int(z))
        mask = np.logical_not(mask)
        maskedSlice = ma.array(mySlice, mask=mask)
        pylab.imshow(maskedSlice, cmap=pylab.gray())
        pylab.show()
    
#     mask = nodule.regions.getRegionMasksSphere(cc)
#     for z in range(0, mask.shape[2]):
#         mask2D = mask[:,:,z]
#         mySlice = reader.dfr.getSlicePixelsRescaled(int(z))
#         mask2D = np.logical_not(mask2D)
#         maskedSlice = ma.array(mySlice, mask=mask2D)
#         pylab.imshow(maskedSlice, cmap=pylab.gray())
#         pylab.show()
        
        