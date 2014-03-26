import dicom
import numpy
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

        #assuming properties are the same for all slices
        if self.Slices[0].ImageOrientationPatient != [1, 0, 0, 0, 1, 0]:
            raise Exception("Unsupported image orientation")
            #deze richtingscosinussen kunnen eventueel ook in world matrix verwerkt worden
        
        if self.Slices[0].PatientPosition != "FFS":
            raise Exception("Unsupported patient position")
        
        if self.Slices[0].SliceLocation != self.Slices[0].ImagePositionPatient[2]:
            raise Exception("SliceLocation != ImagePositionZ")

    def getMinZ(self):
        return min([ s.ImagePositionPatient[2] for s in self.Slices])
    
    def getMaxZ(self):
        return max([ s.ImagePositionPatient[2] for s in self.Slices])

    #world = M * voxel
    def getWorldMatrix(self):
        ds = self.Slices[0];
        return numpy.matrix([[ds.PixelSpacing[0], 0, 0, ds.ImagePositionPatient[0] - ds.PixelSpacing[0]/2],
                             [0, ds.PixelSpacing[1], 0, ds.ImagePositionPatient[1] - ds.PixelSpacing[1]/2],
                             [0, 0, ds.SliceThickness,  self.getMinZ() - ds.SliceThickness/2],
                             [0, 0, 0, 1]])
#</class>
class CoordinateConverter:
    def __init__(self, matrix): 
        self.Matrix = matrix;
    
    #TODO remove references to self.Slices 
    def getPixelZ(self, worldZ):
        dz = self.Slices[0].SliceThickness;
        return (worldZ - self.getMinZ() + dz/2) / dz

    def getWorldZ(self, pixelZ):
        dz = self.Slices[0].SliceThickness;
        return pixelZ * dz + self.getMinZ() - dz/2
       
    #TODO implement using matrix
    def getPixelVector(self, worldVector):
        pass
    
    def getWorldVector(self, pixelVector):
        pass

#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
#dfr = DicomFolderReader(myPath)