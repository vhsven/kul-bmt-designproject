import re
import dicom
import numpy as np
from os import listdir, walk
from os.path import isfile, join
from CoordinateConverter import CoordinateConverter
from collections import deque
from XmlAnnotationReader import XmlAnnotationReader

class DicomFolderReader:
    @staticmethod
    def findAllPaths(rootPath="../data/LIDC-IDRI", maxPaths=99999):
        """Returns the list of lowest possible subdirectories under rootPath that have files in them."""
        count = 0
        for dirPath, dirs, files in walk(rootPath):
            if count < maxPaths and files and not dirs:
                count += 1
                yield dirPath
                
    @staticmethod
    def findPathByID(rootPath, setID):
        for dirPath, dirs, files in walk(rootPath):
            if files and not dirs:
                m = re.search('LIDC-IDRI-(\d\d\d\d)', dirPath)
                mySetID = int(m.group(1))
                if mySetID == setID:
                    return dirPath
                    
    @staticmethod
    def findPathsByID(rootPath, setIDs):
        for dirPath, dirs, files in walk(rootPath):
            if files and not dirs:
                m = re.search('LIDC-IDRI-(\d\d\d\d)', dirPath)
                mySetID = int(m.group(1))
                if mySetID in setIDs:
                    yield dirPath
        
    @staticmethod
    def create(rootPath, setID):
        myPath = DicomFolderReader.findPathByID(rootPath, setID)
        return DicomFolderReader(myPath, True)
    
    def getSetID(self):
        m = re.search('LIDC-IDRI-(\d\d\d\d)', self.Path)
        return int(m.group(1))
        
    def __init__(self, myPath, compress=False):
        self.Path = myPath
        self.Data = None
        self.Slices = deque()
        self.RescaleSlope = 1
        self.RescaleIntercept = 0
        self.IsCompressed = False
        
        myFiles = [ join(myPath, f) for f in listdir(myPath) if isfile(join(myPath, f)) and f.lower().endswith(".dcm") ]
        try:
            for myFile in myFiles:
                self.Slices.append(dicom.read_file(myFile))
        except Exception as e:
            print("DICOM parsing failed for file '{1}': {0}".format(e, myFile))
            exit(1)
        
        #if you get an error here, make sure the data folder only contains CT scans, not RX.
        self.Slices = sorted(self.Slices, key=lambda s: s.SliceLocation) #silly slices are not sorted yet
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
        
        if compress:
            self.compress()

    def __del__(self):
        del self.Path 
        del self.Slices
        del self.RescaleSlope
        del self.RescaleIntercept
        del self.IsCompressed
        del self.Data
    
    def __str__(self):
        return "DicomFolderReader('{}') with {} slices.".format(self.Path, self.getNbSlices())
    
    def compress(self):
        """Deletes data from DICOM header and consolidates it in a 3D array."""
        
        if self.IsCompressed:
            raise ValueError("Already compressed")
        
        shape = self.getVolumeShape()
        self.Data = np.zeros(shape, dtype=np.int16)
        for z in range(0, self.getNbSlices()):
            self.Data[:,:,z] = self.getSliceDataRescaled(z)
            del self.Slices[z]._pixel_array
            del self.Slices[z].PixelData
        
        self.IsCompressed = True
        
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
    
    def getAnnotationReader(self):
        return XmlAnnotationReader(self.Path, self.getCoordinateConverter())
    
    def getNbSlices(self):
        return len(self.Slices)
    
    def getSliceShape(self):
        return self.getSliceData(0).shape
    
    def getVolumeShape(self):
        return self.getSliceData(0).shape + (self.getNbSlices(),)
    
    def getVoxelShape(self):
        ds = self.Slices[0]
        return ds.PixelSpacing[0], ds.PixelSpacing[1], ds.SliceThickness
        
    def getSliceData(self, index):
        if self.IsCompressed:
            return self.Data[:,:,index]
        else:
            return self.Slices[index].pixel_array
    
    def getSliceDataRescaled(self, index):
        return self.getSliceData(index) * self.RescaleSlope - self.RescaleIntercept
    
    def getVolumeData(self):
        if not self.IsCompressed:
            print("Compressing first...")
            self.compress()
         
        return self.Data
    
    def printInfo(self, prefix=""):
        reader = self.getAnnotationReader()
        print("")
        print(prefix + "Voxel shape:", self.getVoxelShape())
        print(prefix + "Nodule positions and radii in dataset: ", self.getSetID())
        for c, r in reader.getNodulePositions(max):
            print "{}{} -> {}".format(prefix, c, r)
        print("")