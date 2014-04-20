import dicom
import scipy
import numpy as np
import numpy.ma as ma
from skimage.morphology import reconstruction, binary_opening, binary_erosion
from os import listdir, walk
from os.path import isfile, join
from CoordinateConverter import CoordinateConverter
from Constants import DEFAULT_THRESHOLD, DEFAULT_WINDOW_SIZE #, ZOOM_FACTOR_3D

class DicomFolderReader:
    Slices = []
    
    @staticmethod
    def findPaths(rootPath, maxPaths=99999):
        """Returns the list of lowest possible subdirectories under rootPath that have files in them."""
        count = 0
        for dirPath, dirs, files in walk(rootPath):
            if count < maxPaths and files and not dirs:
                count += 1
                yield dirPath
                
    @staticmethod
    def findPath(rootPath, index):
        return list(DicomFolderReader.findPaths(rootPath))[index-1] #LIDC-IDRI-0001 has index 0
    
    def __init__(self, myPath):
        myFiles = [ join(myPath, f) for f in listdir(myPath) if isfile(join(myPath, f)) and f.lower().endswith(".dcm") ]
        self.Slices = []
        try:
            for myFile in myFiles:
                self.Slices.append(dicom.read_file(myFile))
        except Exception as e:
            print("DICOM parsing failed for file '{1}': {0}".format(e, myFile))
            exit(1)
        
        self.Path = myPath        
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
        
        if self.getSliceShape() != (512, 512):
            raise Exception("Unsupported slice dimensions: {}".format(self.getSliceShape()))
        
        # check whether all slices have the same transform params
        #assert sum([s.RescaleSlope for s in self.Slices]) == self.Slices[0].RescaleSlope * len(self.Slices)
        #assert sum([s.RescaleIntercept for s in self.Slices]) == self.Slices[0].RescaleIntercept * len(self.Slices)

    def __del__(self):
        del self.Path 
        del self.Slices
        del self.NbSlices
        del self.Masks
        del self.RescaleSlope
        del self.RescaleIntercept
    
    def __str__(self):
        return "DicomFolderReader('{}') with {} slices.".format(self.Path, self.getNbSlices())
        
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
        h,w,d = self.getVolumeShape()
        #h = int(h * ZOOM_FACTOR_3D)
        #w = int(w * ZOOM_FACTOR_3D)
        voxels = np.zeros((h,w,d), dtype=np.int16)
        for index in range(0, self.getNbSlices()):
            data = self.getSlicePixelsRescaled(index)
            #data = scipy.ndimage.zoom(data, ZOOM_FACTOR_3D)
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