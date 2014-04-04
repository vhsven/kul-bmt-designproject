'''
Created on 3-apr.-2014

@author: Eigenaar
'''
import pylab
import numpy.ma as ma
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

############################################
#### step 2: prepare data

BIN_SIZE = 16
threshold = 1500

# transform the pixel grey values to HU units
intercept = int(ds.RescaleIntercept) # found in dicom header at (0028,1052)
slope = int(ds.RescaleSlope) # found in dicom header at (0028,1053)
HU = data * slope - intercept

# apply a mask to the image to exclude the pixels outside the thorax in the image
minI = HU.min()
maxI = HU.max()
print("rescaled grey levels: {} - {}".format(minI, maxI))
thoraxMask = ma.masked_equal(HU, minI)
minI = thoraxMask.min() # find the new minimum inside mask region

HUshift = HU

if minI != 0: #shift intensities so that minI = 0
    HUshift -= minI
    maxI -= minI
    minI = 0

getthorax = ma.masked_outside(HUshift, BIN_SIZE*0 ,  BIN_SIZE*5)  #  0 -   80
pylab.plot()
pylab.imshow(getthorax, cmap=pylab.gray())
pylab.show()

# get inverse of mask
getthoraxI=~getthorax.mask
pylab.plot()
pylab.imshow(getthoraxI, cmap=pylab.gray())
pylab.show()

# apply again thoraxMask
combinedMask1 = ma.mask_or(ma.getmask(thoraxMask), ma.getmask(getthorax))
pylab.plot()
pylab.imshow(combinedMask1, cmap=pylab.gray())
pylab.show()

combinedMask1 = ma.array(HU, mask=combinedMask1) #apply on matrix

pylab.plot()
pylab.imshow(combinedMask1, cmap=pylab.gray())
pylab.show()

nonLungMask = ma.masked_greater(HU, threshold)
combinedMask = ma.mask_or(ma.getmask(thoraxMask), ma.getmask(nonLungMask))
combinedMask = ma.array(HU, mask=combinedMask) #apply on matrix
 
pylab.subplot(1, 2, 1)
pylab.imshow(thoraxMask, cmap=pylab.gray())
 
pylab.subplot(1, 2, 2)
pylab.imshow(combinedMask, cmap=pylab.gray())
pylab.show()
 
### make feature vector
featurevector=[]


#featurevector[1]=greyvalue (mean, variance, thresholding) (2D Keshani et al)
# grey value

    
#featurevector[2]=intensity of that grey value

#featurevector[3]= 3D averaging (Keshani et al)

#featurevector[4]= position of voxel (regarding the lung wall?)

#featurevector[5]= window: trek links/R af van rechts/L -> verschillende groottes window

#featurevector[6]= window: divide left/R by right/L

#featurevector[7]= window: trek boven/onder af van onder/boven -> verschillende groottes window

#featurevector[8]= window: divide above/below by below/above

#featurevector[9]= features Ozekes and Osman (2010)

#featurevector[10]= edges

#featurevector[11]=uitgerektheid/compact

#featurevector[12]=convolution filters (filter banks): law filters (p296 ev), gabor filters, eigenfilters

# featurevector[13]= afstand tot andere nodules (lichte pixels)

#featurevector[14]= Canny edge detection

# featurevector[10]= gradienten (slide 39 van feature selection pdf)

# featurevector[10]= Kadir and Brady algorithm (entropy)

# featurevector[10]= histogram distances (Manhattan distance, eucledian distance, max distance)
    # deel tekening op in kleine gebieden/ histogrammen en bekijk daar histogram distances ofzo
    
# featurevector[10]= texture analysis -> statistical (mean, variance, skewness, kurtosis) (slide 28 van feature selection pdf)
    # autocorrelations
    
