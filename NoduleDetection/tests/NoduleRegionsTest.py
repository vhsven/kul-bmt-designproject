import pylab as pl
import numpy as np
from XmlAnnotationReader import XmlAnnotationReader
from DicomFolderReader import DicomFolderReader

def yieldMaxRadii():
    for myPath in DicomFolderReader.findAllPaths("../data/LIDC-IDRI"):
        dfr = DicomFolderReader(myPath, False)
        cc = dfr.getCoordinateConverter()
        reader = XmlAnnotationReader(myPath, cc)
        
        for nodule in reader.Nodules:
            _, regionRs = nodule.Regions.getRegionCenters()
            r = max(regionRs.values())
            r *= dfr.getVoxelShape()[0] #convert to mm
            #if r < 5:
            #    print dfr.getSetID()
            yield r
            
radii = np.array(list(yieldMaxRadii()))

print len(radii) #85

#2.6799464451 4.72682497817 6.32230086901 3.82981531547 18.3922190848
print radii.min(), np.median(radii), radii.mean(), radii.std(), radii.max()

pl.hist(radii)
pl.title("Nodule Radius")
pl.xlabel("Radius (mm)")
pl.ylabel("Count")
pl.show()

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
            
            
