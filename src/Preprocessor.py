import sys
import dicom
import pylab as pl
import numpy as np
import numpy.ma as ma
from skimage.morphology import reconstruction, binary_dilation
from Constants import DEFAULT_THRESHOLD
from DicomFolderReader import DicomFolderReader

class Preprocessor:
#     def processSlice(self, index, threshold, erosionWindow):
#         HU = self.getSlicePixelsRescaled(index)
#         
#         # select pixels in thorax wall
#         masked = ma.masked_greater(HU, threshold)
#         
#         # opening to remove small structures
#         newmask = binary_opening(masked.mask, selem=np.ones((9,9)))
#     
#         # reconstruction to fill lungs (figure out how this works) 
#         seed = np.copy(newmask)
#         seed[1:-1, 1:-1] = newmask.max()
#         newmask = reconstruction(seed, newmask, method='erosion').astype(np.int)
#         
#         # erode thorax walls slightly (no nodules in there)
#         newmask = binary_erosion(newmask, selem=np.ones((erosionWindow,erosionWindow))).astype(np.int)
#         
#         masked = ma.array(HU, mask=np.logical_not(newmask))
#         
#         return masked
    
    @staticmethod
    def getThresholdMask(data):
        sys.stdout.write("Performing initial segmentation per slice")
        
        mask3D = data > DEFAULT_THRESHOLD #select soft tissue (thorax wall etc.)
        del data
        
        for z in range(0, mask3D.shape[2]):
            sys.stdout.write('.')
            
            # reconstruction to fill lungs (figure out how this works) 
            seed = np.copy(mask3D[:,:,z])
            seed[1:-1, 1:-1] = mask3D[:,:,z].max()
            mask3D[:,:,z] = reconstruction(seed, mask3D[:,:,z], method='erosion').astype(np.bool) - mask3D[:,:,z]
            
            mask3D[:,:,z] = binary_dilation(mask3D[:,:,z], selem=np.ones((29,29))).astype(np.bool)            
            
            #masked = ma.array(HU, mask=~mask3D[:,:,z])
            #pl.imshow(masked, cmap=pl.gray())
            #pl.title("Dilated")
            #pl.show()
            
        print("")
        return mask3D
    
    @staticmethod
    def loadThresholdMask(setID):
        try:
            mask = dicom.read_file("../data/LIDC-Masks/img{}.dcm".format(setID)).pixel_array.astype(bool)
            mask = np.rollaxis(mask, 0, 3)
            z = mask.shape[2]
            
            #make sure mask is false at top and bottom to prevent out of bounds errors
            mask[:,:,0:6] = False
            mask[:,:,z-6:z] = False
            #print("\tLoaded lung mask for dataset {}.".format(setID))
            return mask
        except:
            raise ValueError("Lung mask not available for given dataset.")
    
    @staticmethod
    def checkMask(setID, mySlice, rootPath="../data/LIDC-IDRI"):
        dfr = DicomFolderReader.create(rootPath, setID)
        print dfr.Path
        vData = dfr.getVolumeData()
        vMask = Preprocessor.loadThresholdMask(setID)
        
        sData = vData[...,mySlice]
        sMask = vMask[...,mySlice]
        masked = ma.masked_array(sData, mask=~sMask)
        
        pl.imshow(masked, cmap=pl.gray())
        pl.show()

if __name__ == "__main__":     
    Preprocessor.checkMask(46, 89)
        
