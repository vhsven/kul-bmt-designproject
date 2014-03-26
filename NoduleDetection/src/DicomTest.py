#!/usr/local/bin/python2.7
# encoding: utf-8

import dicom
import numpy
import matplotlib.pylab as plot
import matplotlib.cm as cm

# to find current working directory:
# os.getcwd()
# voxel(i,j) = pixel_data[j, i]

#ds = dicom.read_file("../data/000000.dcm")
ds = dicom.read_file(r"C:\Users\Sven\Desktop\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000\000000.dcm") #raw string
ds132 = dicom.read_file(r"C:\Users\Sven\Desktop\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000\000132.dcm")

#plot.imshow(ds.pixel_array, cmap=plot.gray())
#plot.show()

if ds.ImageOrientationPatient != [1, 0, 0, 0, 1, 0]:
    raise Exception("Unsupported image orientation")
    #deze richtingscosinussen kunnen eventueel ook in world matrix verwerkt worden

if ds.PatientPosition != "FFS":
    raise Exception("Unsupported patient position")

if ds.SliceLocation != ds.ImagePositionPatient[2]:
    raise Exception("SliceLocation != ImagePositionZ")

#use ds.dir("keyword") to search fields

#print(ds)

#world = M * voxel
def getWorldMatrix(ds):
    return numpy.matrix([[ds.PixelSpacing[0], 0, 0, ds.ImagePositionPatient[0]],
                         [0, ds.PixelSpacing[1], 0, ds.ImagePositionPatient[1]],
                         [0, 0, ds.SliceThickness,  ds.ImagePositionPatient[2]],
                         [0, 0, 0, 1]])

#TODO waar haalt MVL die -341.25?

def getPixelZ(worldZ):
    return (worldZ - -341.25) / ds.SliceThickness

def getWorldZ(pixelZ):
    return pixelZ * ds.SliceThickness + -341.25

print(getWorldMatrix(ds))
print(getWorldMatrix(ds132))
print(getPixelZ(-125.0))
print(getWorldZ(86.5))

#0.703125  0        0   -166.3515625
#0         0.703125 0   -172.0515595
#0         0        2.5 -341.25
#0         0        0    1