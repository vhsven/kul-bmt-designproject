'''
Created on 9-apr.-2014

@author: Eigenaar
'''
import pylab
import numpy as np
import numpy.ma as ma
from DicomFolderReader import DicomFolderReader
import collections
from numpy import linalg
import math



###########################################
### step 1: import data

# myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
# dfr = DicomFolderReader(myPath)
# ds = dfr.Slices[50]
# data = ds.pixel_array #voxel(i,j) is pixel(j,i) -> so one voxel is one pixel (http://nipy.org/nibabel/dicom/dicom_orientation.html)
# print(data)
# #show image
# pylab.imshow(ds.pixel_array, cmap=pylab.gray())
# pylab.show()

############################################
#### step 2: prepare data

# select thorax

###########################################
#### step 3: make 2D feature vector
featurevector=[]

#featurevector[1]= position of pixel in 2D slice
# x and y are the pixelcoordinates of certain position in image
x=5
y=5


#featurevector[2]=greyvalue
def greyvaluecharateristic(x,y,windowrowvalue,data):
    # windowrowvalue should be odd number (3,5,7...)
    
    # grey value
    greyvalue=data[x,y]
    
    # mean value (square windowrowvalue x windowrowvalue)
    valup=math.ceil(windowrowvalue/2)
    valdown=math.floor(windowrowvalue/2)
    
    D=data[x-valdown:x+valup,y-valdown:y+valup]
    Dsom1=sum(D[:,:])
    Dsom2=sum(Dsom1[:])
    M=Dsom2/(windowrowvalue**2)
    
    # standard deviation
    V=D.std()
    
    return greyvalue, M, V

 
#featurevector[3]= prevalence of that grey value
def greyvaluefrequency (x,y,data):
    m,n=data.shape
    mydata = np.reshape(data, (m*n))
    #import collections
    counter=collections.Counter(mydata)
    freqvalue=counter[mydata[m*x+y]]
    
    return freqvalue


#featurevector[4]=  frobenius norm pixel to center 2D image
def forbeniusnorm (x,y):
    # slice is 512 by 512: b is center
    xb=256
    yb=256
    a = np.array((x,y))
    b = np.array((xb,yb))
    dist = np.linalg.norm(a-b)
    
    # remark: easy to extend to 3D
    
    return dist


#featurevector[5]= window: substraction L R
def substractwindowLR(x,y,windowrowvalue,data):
    valup=math.ceil(windowrowvalue/2)
    valdown=math.floor(windowrowvalue/2)
    
    windowD=data[x-valdown:x+valup,y-valdown:y+valup]
    
    # calculate 'gradients' by substraction 
    leftrow=windowD[:,0]
    rightrow=windowD[:,(windowrowvalue-1)]
    gradLR=rightrow-leftrow
    
    # calculate 'gradients' by substraction 
    gradRL=leftrow-rightrow
    
    return gradLR, gradRL


#featurevector[6]= window: divide left/R by right/L
def dividewindowLR(x,y,windowrowvalue,data):
    valup=math.ceil(windowrowvalue/2)
    valdown=math.floor(windowrowvalue/2)
    
    windowD=data[x-valdown:x+valup,y-valdown:y+valup]
    
    # calculate 'gradients' by division 
    leftrow=windowD[:,0]
    rightrow=windowD[:,(windowrowvalue-1)]
    divLR=leftrow/rightrow
    
    # calculate 'gradients'
    divRL=rightrow/leftrow
    
    return divLR, divRL   


#featurevector[7]= window: substraction Up Down
def substractwindowUD(x,y,windowrowvalue,data):
    valup=math.ceil(windowrowvalue/2)
    valdown=math.floor(windowrowvalue/2)
    
    windowD=data[x-valdown:x+valup,y-valdown:y+valup]
    
    # calculate 'gradients' by substraction 
    toprow=windowD[0,:]
    bottomrow=windowD[(windowrowvalue-1), :]
    gradUD=toprow-bottomrow
    
    # calculate 'gradients' by substraction 
    gradDU=bottomrow-toprow
    
    return gradUD, gradDU


#featurevector[8]= window: divide above/below by below/above
def dividewindowUD(x,y,windowrowvalue,data):
    valup=math.ceil(windowrowvalue/2)
    valdown=math.floor(windowrowvalue/2)
    
    windowD=data[x-valdown:x+valup,y-valdown:y+valup]
    
    # calculate 'gradients' by substraction 
    toprow=windowD[0,:]
    bottomrow=windowD[(windowrowvalue-1), :]
    divUD=toprow/bottomrow
    
    # calculate 'gradients' by substraction 
    divDU=bottomrow/toprow
    
    return divUD, divDU

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
    
#featurevector[3]= 3D averaging (Keshani et al)

#sklearn: image feature 2