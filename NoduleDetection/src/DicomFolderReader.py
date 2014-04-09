import dicom
import numpy as np
#import matplotlib.pylab as plot
#import matplotlib.cm as cm
#plot.imshow(ds.pixel_array, cmap=plot.gray())
#plot.show()
from os import listdir
from os.path import isfile, join

# to find current working directory:
# os.getcwd()
# voxel(i,j) = pixel_data[j, i]

#TODO split coordinate conversion functions from reader functions
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
        
    def getNbSlices(self):
        return len(self.Slices)
    
    def getSliceData(self, index):
        return self.Slices[index]
    
    def getSlicePixels(self, index):
        return self.Slices[index].pixel_array
    
    def getSlicePixelsRescaled(self, index):
        return self.Slices[index].pixel_array * self.RescaleSlope - self.RescaleIntercept