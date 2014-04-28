import sys
import dicom
import numpy as np
from skimage.morphology import reconstruction, binary_dilation
from Constants import DEFAULT_THRESHOLD

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
        mask = dicom.read_file("../data/LIDC-Masks/img{}.dcm".format(setID)).pixel_array.astype(bool)
        mask = np.rollaxis(mask, 0, 3)
        print("Loaded mask for dataset {}.".format(setID))
        return mask