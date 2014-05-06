import pylab
import numpy as np
import numpy.ma as ma
from XmlAnnotationReader import XmlAnnotationReader
from DicomFolderReader import DicomFolderReader

for myPath in DicomFolderReader.findPaths("../data/LIDC-IDRI"):
    
    dfr = DicomFolderReader(myPath)
    #m,n,_ = dfr.getVolumeShape()
    cc = dfr.getCoordinateConverter()
    reader = XmlAnnotationReader(myPath, cc)
    
    if len(reader.Nodules) == 0:
        print myPath
    else:
        print "."
        
    # for c,r2 in reader.getNodulePositions():
    #     print(c,r2)
    
#     for nodule in reader.Nodules:
#         print(nodule.ID)
#         #nodule.Regions.printRegions()
#         masks, c, r = nodule.Regions.getRegionMasksCircle(m,n,0.33)
#         #paths, masks = nodule.Regions.getRegionMasksPolygon(m,n)
#            
#         for z in masks.keys():
#             mask = masks[z]
#             mySlice = reader.dfr.getSliceDataRescaled(int(z))
#             mask = np.logical_not(mask)
#             maskedSlice = ma.array(mySlice, mask=mask)
#             pylab.imshow(maskedSlice, cmap=pylab.gray())
#             pylab.show()
        
    #     mask = nodule.Regions.getRegionMasksSphere(cc)
    #     for z in range(0, mask.shape[2]):
    #         mask2D = mask[:,:,z]
    #         mySlice = reader.dfr.getSliceDataRescaled(int(z))
    #         mask2D = np.logical_not(mask2D)
    #         maskedSlice = ma.array(mySlice, mask=mask2D)
    #         pylab.imshow(maskedSlice, cmap=pylab.gray())
    #         pylab.show()
            
            
