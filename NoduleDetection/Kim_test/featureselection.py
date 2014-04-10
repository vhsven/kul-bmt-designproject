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
#x
#y


#featurevector[2]=greyvalue
def greyvaluecharateristic(x,y,windowrowvalue,data):
    # windowrowvalue should be odd number (3,5,7...)
    
    # grey value
    greyvalue=data[x,y]
    
    # square windowrowvalue x windowrowvalue
    valup=math.ceil(windowrowvalue/2)
    valdown=math.floor(windowrowvalue/2)
    
    windowD=data[x-valdown:x+valup,y-valdown:y+valup]
    
    #reshape window into array
    h,w=windowD.shape()
    arrayD = np.reshape(windowD, (h*w))
    
    # mean and variance
    M=arrayD.mean()
    V=arrayD.var()
        
    x = range(w)
    y = range(h)


    #calculate projections along the x and y axes
    yp = np.sum(windowD,axis=1)
    xp = np.sum(windowD,axis=0)

    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)

    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2

    sx = np.sqrt( np.sum(x2*xp)/np.sum(xp) )
    sy = np.sqrt( np.sum(y2*yp)/np.sum(yp) )

    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3

    skx = np.sum(xp*x3)/(np.sum(xp) * sx**3)
    sky = np.sum(yp*y3)/(np.sum(yp) * sy**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    kx = np.sum(xp*x4)/(np.sum(xp) * sx**4)
    ky = np.sum(yp*y4)/(np.sum(yp) * sy**4)
    
    #autocorrelation
    result = np.correlate(arrayD, arrayD, mode='full')
    autocorr=result[result.size/2:]


    return greyvalue,M,V,cx,cy,sx,sy,skx,sky,kx,ky,autocorr
    
   
 
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
def windowLR(x,y,windowrowvalue,data):
    valup=math.ceil(windowrowvalue/2)
    valdown=math.floor(windowrowvalue/2)
    
    windowD=data[x-valdown:x+valup,y-valdown:y+valup]
    
    # calculate 'gradients' by substraction 
    leftrow=windowD[:,0]
    rightrow=windowD[:,(windowrowvalue-1)]
    meanL=leftrow.mean()
    meanR=rightrow.mean()
    gradLR=rightrow-leftrow
    gradmeanLR=meanR-meanL
    
    # calculate 'gradients' by division
    divmeanLR=meanR/meanL    
    divLR=leftrow/rightrow
    
       
    return gradLR, gradmeanLR, divLR, divmeanLR


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

#featurevector[9]= 3Dfeatures Ozekes and Osman (2010)

#featurevector[10]= edges

#featurevector[11]=uitgerektheid/compact

#featurevector[12]=convolution filters (filter banks)

#law filters (p296 ev)
#gabor filters
#eigenfilters

# featurevector[13]= 

#featurevector[14]= Canny edge detection

# featurevector[10]= gradienten (slide 39 van feature selection pdf)

# featurevector[10]= Kadir and Brady algorithm (entropy)

# featurevector[10]= histogram distances (Manhattan distance, eucledian distance, max distance)
    # deel tekening op in kleine gebieden/ histogrammen en bekijk daar histogram distances ofzo
    

    
#featurevector[3]= 3D averaging (Keshani et al)

#sklearn: image feature 2