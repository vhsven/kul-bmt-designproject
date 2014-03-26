#!/usr/local/bin/python2.7
# encoding: utf-8

import dicom
import numpy
import matplotlib.pylab as plot
import matplotlib.cm as cm
from os import listdir
from os.path import isfile, join

# to find current working directory:
# os.getcwd()
# voxel(i,j) = pixel_data[j, i]

#ds = dicom.read_file("../data/000000.dcm")
myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
myFiles = [ join(myPath, f) for f in listdir(myPath) if isfile(join(myPath, f)) and f.lower().endswith(".dcm") ]
slices = []
try:
    for myFile in myFiles:
        slices.append(dicom.read_file(myFile))
except Exception as e:
    print("DICOM parsing failed for file '{1}': {0}".format(e, myFile))
    exit(1)

slices = sorted(slices, key=lambda s: s.SliceLocation) #silly slices are not sorted yet
#for mySlice in slices:
#    print(mySlice.SliceLocation)
    
#ds = dicom.read_file(r"C:\Users\Sven\Desktop\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000\000000.dcm") #raw string
#ds132 = dicom.read_file(r"C:\Users\Sven\Desktop\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000\000132.dcm")

#plot.imshow(ds.pixel_array, cmap=plot.gray())
#plot.show()

#assuming properties are the same for all slices
if slices[0].ImageOrientationPatient != [1, 0, 0, 0, 1, 0]:
    raise Exception("Unsupported image orientation")
    #deze richtingscosinussen kunnen eventueel ook in world matrix verwerkt worden

if slices[0].PatientPosition != "FFS":
    raise Exception("Unsupported patient position")

if slices[0].SliceLocation != slices[0].ImagePositionPatient[2]:
    raise Exception("SliceLocation != ImagePositionZ")

#use ds.dir("keyword") to search fields

#print(ds)

def getMinZ(slices):
    return min([ s.ImagePositionPatient[2] for s in slices])

#world = M * voxel
def getWorldMatrix(slices):
    ds = slices[0];
    return numpy.matrix([[ds.PixelSpacing[0], 0, 0, ds.ImagePositionPatient[0] - ds.PixelSpacing[0]/2],
                         [0, ds.PixelSpacing[1], 0, ds.ImagePositionPatient[1] - ds.PixelSpacing[1]/2],
                         [0, 0, ds.SliceThickness,  getMinZ(slices) - ds.SliceThickness/2],
                         [0, 0, 0, 1]])

#TODO waar haalt MVL die -341.25?

def getPixelZ(slices, worldZ):
    dz = slices[0].SliceThickness;
    return (worldZ - getMinZ(slices) - dz/2) / dz

def getWorldZ(slices, pixelZ):
    dz = slices[0].SliceThickness;
    return pixelZ * dz + getMinZ(slices) + dz/2

print(getWorldMatrix(slices))
print(getPixelZ(slices, -121.25))
print(getWorldZ(slices, 87))

#Matrix from MeVisLab:
#0.703125  0        0   -166.3515625
#0         0.703125 0   -172.0515595
#0         0        2.5 -341.25
#0         0        0    1