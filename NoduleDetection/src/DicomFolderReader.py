import dicom
import numpy as np
import numpy.ma as ma
from skimage.morphology import reconstruction, binary_opening, binary_erosion
from os import listdir
from os.path import isfile, join
from CoordinateConverter import CoordinateConverter
from Constants import *

class DicomFolderReader:
    Slices = []
    
    def __init__(self, myPath):
        myFiles = [ join(myPath, f) for f in listdir(myPath) if isfile(join(myPath, f)) and f.lower().endswith(".dcm") ]
        try:
            for myFile in myFiles:
                self.Slices.append(dicom.read_file(myFile))
        except Exception as e:
            print("DICOM parsing failed for file '{1}': {0}".format(e, myFile))
            exit(1)
                
        self.Slices = sorted(self.Slices, key=lambda s: s.SliceLocation) #silly slices are not sorted yet
        self.NbSlices = len(self.Slices)
        self.Masks = [None] * self.NbSlices
        self.RescaleSlope = int(self.Slices[0].RescaleSlope)
        self.RescaleIntercept = int(self.Slices[0].RescaleIntercept)
        
        #assuming properties are the same for all slices
        if self.Slices[0].ImageOrientationPatient != [1, 0, 0, 0, 1, 0]:
            raise Exception("Unsupported image orientation")
            #deze richtingscosinussen kunnen eventueel ook in world matrix verwerkt worden
        
        if self.Slices[0].PatientPosition != "FFS":
            raise Exception("Unsupported patient position")
        
        if self.Slices[0].SliceLocation != self.Slices[0].ImagePositionPatient[2]:
            raise Exception("SliceLocation != ImagePositionZ")
        
        # check whether all slices have the same transform params
        #assert sum([s.RescaleSlope for s in self.Slices]) == self.Slices[0].RescaleSlope * len(self.Slices)
        #assert sum([s.RescaleIntercept for s in self.Slices]) == self.Slices[0].RescaleIntercept * len(self.Slices)

    def getMinZ(self):
        return min([s.ImagePositionPatient[2] for s in self.Slices])
    
    def getMaxZ(self):
        return max([s.ImagePositionPatient[2] for s in self.Slices])

    #world = M * voxel
    def getWorldMatrix(self):
        ds = self.Slices[0];
        return np.matrix([[ds.PixelSpacing[0], 0, 0, ds.ImagePositionPatient[0] - ds.PixelSpacing[0]/2],
                             [0, ds.PixelSpacing[1], 0, ds.ImagePositionPatient[1] - ds.PixelSpacing[1]/2],
                             [0, 0, ds.SliceThickness,  self.getMinZ() - ds.SliceThickness/2],
                             [0, 0, 0, 1]])
        
    def getCoordinateConverter(self):    
        return CoordinateConverter(self.getWorldMatrix())
    
    def getNbSlices(self):
        return len(self.Slices)
    
    def getSliceShape(self):
        return self.getSlicePixels(0).shape
    
    def getVolumeShape(self):
        return self.getSlicePixels(0).shape + (self.getNbSlices(),)
    
    def getVoxelShape(self):
        ds = self.Slices[0]
        return ds.PixelSpacing[0], ds.PixelSpacing[1], ds.SliceThickness
        
    def getSliceData(self, index):
        return self.Slices[index]
    
    def getSlicePixels(self, index):
        return self.Slices[index].pixel_array
    
    def getSlicePixelsRescaled(self, index):
        return self.Slices[index].pixel_array * self.RescaleSlope - self.RescaleIntercept
    
    def getVolumeData(self):
        voxels = np.zeros(self.getVolumeShape())
        for index in range(0, self.getNbSlices()):
            data = self.getSlicePixelsRescaled(index)
            voxels[:,:,index] = data
            
        return voxels
    
    def processSlice(self, index, threshold, erosionWindow):
        HU = self.getSlicePixelsRescaled(index)
        
        # select pixels in thorax wall
        masked = ma.masked_greater(HU, threshold)
        
        # opening to remove small structures
        newmask = binary_opening(masked.mask, selem=np.ones((9,9)))
    
        # reconstruction to fill lungs (figure out how this works) 
        seed = np.copy(newmask)
        seed[1:-1, 1:-1] = newmask.max()
        newmask = reconstruction(seed, newmask, method='erosion').astype(np.int)
        
        # erode thorax walls slightly (no nodules in there)
        newmask = binary_erosion(newmask, selem=np.ones((erosionWindow,erosionWindow))).astype(np.int)
        
        masked = ma.array(HU, mask=np.logical_not(newmask))
        
        return masked
    
    def getMaskedSlice(self, index):
        if self.Masks[index] == None:
            self.Masks[index] = self.processSlice(index, DEFAULT_THRESHOLD, DEFAULT_WINDOW_SIZE)
        
        return self.Masks[index]