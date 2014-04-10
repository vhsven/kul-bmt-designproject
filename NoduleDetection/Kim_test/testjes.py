'''
Created on 9-apr.-2014

@author: Eigenaar
'''
import math

import pylab
import numpy.ma as ma
import numpy as np
from DicomFolderReader import DicomFolderReader 

###########################################
### step 1: import data

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
dfr = DicomFolderReader(myPath)
ds = dfr.Slices[50]
data = ds.pixel_array #voxel(i,j) is pixel(j,i) -> so one voxel is one pixel (http://nipy.org/nibabel/dicom/dicom_orientation.html)
#show image
pylab.imshow(ds.pixel_array, cmap=pylab.gray())
pylab.show()


# myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
# dfr = DicomFolderReader(myPath)
# data = dfr.getVolumeData() #voxel(i,j) is pixel(j,i) -> so one voxel is one pixel (http://nipy.org/nibabel/dicom/dicom_orientation.html)
# X,Y,Z = dfr.getVolumeShape()

#####################################################""
######################################################
import matplotlib.pyplot as plt

from skimage.filter.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_int

image=data.view('uint8')

#fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))

# img0 = ax0.imshow(image, cmap=plt.cm.gray)
# ax0.set_title('Image')
# ax0.axis('off')
# plt.colorbar(img0, ax=ax0)

imentropy=entropy(image, disk(5))
print(imentropy)
# img1 = ax1.imshow(imentropy, cmap=plt.cm.jet)
# ax1.set_title('Entropy')
# ax1.axis('off')
# plt.colorbar(img1, ax=ax1)

plt.show()



def image_entropy(img):
    """calculate the entropy of an image"""
    histogram,_ = np.histogram(img,100)
    histogram_length = sum(histogram)

    samples_probability = [float(h) / histogram_length for h in histogram]

    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])

img = data
print image_entropy(img)
