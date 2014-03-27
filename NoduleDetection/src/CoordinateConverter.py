from numpy import linalg as la
import numpy as np

class CoordinateConverter:
    def __init__(self, matrix): 
        self.Matrix = matrix;
        self.Inverse = la.inv(matrix)
    
    def getPixelZ(self, worldZ):
        #dz = self.Slices[0].SliceThickness;
        #return (worldZ - self.getMinZ() + dz/2) / dz
        return self.getPixelVector([0, 0, worldZ, 1])[0,2]

    def getWorldZ(self, pixelZ):
        #dz = self.Slices[0].SliceThickness;
        #return pixelZ * dz + self.getMinZ() - dz/2
        return self.getWorldVector([0, 0, pixelZ, 1])[0,2]

    def getPixelVector(self, worldVector):
        return np.dot(self.Inverse, worldVector)
    
    def getWorldVector(self, pixelVector):
        return np.dot(self.Matrix, pixelVector)