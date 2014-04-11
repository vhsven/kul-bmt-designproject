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
import matplotlib.pyplot as plt
from skimage.filter.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte





###############################################################################
### step 1: import data

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
dfr = DicomFolderReader(myPath)
data = dfr.getVolumeData() # 3D matrix with data

Xsize, Ysize, Zsize = dfr.getVoxelShape() # size of voxel in mm

X,Y,Z = dfr.getVolumeShape() # size of 3D datamatrix in 3D

# REMARK: voxel(i,j) is pixel(j,i)
# one voxel is one pixel (http://nipy.org/nibabel/dicom/dicom_orientation.html)

###############################################################################
#### step 2: prepare data

# select thorax

###############################################################################
#### step 3: make 2D feature vector
featurevector = []

############################################################
#featurevector[1]= position of pixel in 2D slice
############################################################
# x and y are the pixelcoordinates of certain position in image
#position=[x,y,z]


############################################################
#featurevector[2]=greyvalue
############################################################
def greyvaluecharateristic(x,y,z,windowrowvalue,data):
    # windowrowvalue should be odd number (3,5,7...)
    
    # grey value
    greyvalue=data[x,y,z]
    
    # square windowrowvalue x windowrowvalue
    valup=math.ceil(windowrowvalue/2)
    valdown=math.floor(windowrowvalue/2)
    
    windowD=data[x-valdown:x+valup,y-valdown:y+valup,z-valdown:z+valup]
    
    #reshape window into array
    h,w,d=windowD.shape()
    arrayD = np.reshape(windowD, (h*w*d))
    
    # mean and variance
    M=arrayD.mean()
    V=arrayD.var()
    
    x = range(w)
    y = range(h)
    z= range(d)


    #calculate projections along the x and y axes
    zp = np.sum(windowD,axis=2)
    yp = np.sum(windowD,axis=1)
    xp = np.sum(windowD,axis=0)

    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)
    cz = np.sum(z*zp)/np.sum(zp)

    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2
    z2 = (z-cz)**2

    sx = np.sqrt( np.sum(x2*xp)/np.sum(xp) )
    sy = np.sqrt( np.sum(y2*yp)/np.sum(yp) )
    sz = np.sqrt( np.sum(z2*zp)/np.sum(zp) )

    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3
    z3 = (z-cz)**3

    skx = np.sum(xp*x3)/(np.sum(xp) * sx**3)
    sky = np.sum(yp*y3)/(np.sum(yp) * sy**3)
    skz = np.sum(zp*z3)/(np.sum(zp) * sz**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    z4 = (z-cz)**4
    kx = np.sum(xp*x4)/(np.sum(xp) * sx**4)
    ky = np.sum(yp*y4)/(np.sum(yp) * sy**4)
    kz = np.sum(zp*z4)/(np.sum(zp) * sz**4)
    
    #autocorrelation
    result = np.correlate(arrayD, arrayD, mode='full')
    autocorr=result[result.size/2:]


    return greyvalue,M,V,cx,cy,cz,sx,sy,sz,skx,sky,skz,kx,ky,kz,autocorr
    
   
############################################################
#featurevector[3]= prevalence of that grey value
############################################################
def greyvaluefrequency (x,y,z,data):
    m,n,d = data.shape
    mydata = np.reshape(data, (m*n*d))
    #import collections
    counter = collections.Counter(mydata)
    freqvalue = counter[data[x,y,z]]
    
    return freqvalue


############################################################
#featurevector[4]=  frobenius norm pixel to center 2D image
############################################################
def forbeniusnorm (x,y,z):
    # slice is 512 by 512 by numberz: b is center
    xb = 256
    yb = 256
    zb = int[Z/2]
    a = np.array((x,y,z))
    b = np.array((xb,yb,zb))
    dist = np.linalg.norm(a-b)
    
    return dist


############################################################
#featurevector[5]= window: substraction L R
############################################################
def windowLR(x,y,z,windowrowvalue,data):
    valup=math.ceil(windowrowvalue/2)
    valdown=math.floor(windowrowvalue/2)
    
    windowD=data[x-valdown:x+valup,y-valdown:y+valup, z-valdown:z+valup]
    
    # calculate 'gradients' by substraction 
    leftrow=windowD[:,0,:]
    rightrow=windowD[:,(windowrowvalue-1),:]
    meanL=leftrow.mean()
    meanR=rightrow.mean()
    gradLR=rightrow-leftrow
    gradmeanLR=meanR-meanL
    
    # calculate 'gradients' by division
    divmeanLR=meanR/meanL    
    divLR=leftrow/rightrow
    
          
    return gradLR, gradmeanLR, divLR, divmeanLR


############################################################
#featurevector[6]= window: substraction Up Down
############################################################
def substractwindowUD(x, y, z, windowrowvalue, data):
    valup = math.ceil(windowrowvalue/2)
    valdown = math.floor(windowrowvalue/2)
    
    windowD = data[x-valdown:x+valup, y-valdown:y+valup, z-valdown:z+valup]
    
    # calculate 'gradients' by substraction 
    toprow=windowD[0,:,:]
    bottomrow=windowD[(windowrowvalue-1), :, :]
    Tmean=toprow.mean()
    Bmean=bottomrow.mean()
    gradmeanUD=Tmean-Bmean
    gradUD=toprow-bottomrow
    
    # calculate 'gradients' by division
    divUD=toprow/bottomrow
    divmeanUD=Tmean/Bmean
    
    
    return gradUD, gradmeanUD, divmeanUD, divmeanUD


############################################################
# feature[7]= entropy calculation (disk window or entire image)
############################################################
def pixelentropy(x,y,z):
    image=data[:,:,z].view('uint8')
#     fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
    
#     img0 = ax0.imshow(image, cmap=plt.cm.gray)
#     ax0.set_title('Image')
#     ax0.axis('off')
#     plt.colorbar(img0, ax=ax0)
    
    pixel_entropy=entropy(image, disk(5))
#     print(imentropy)
#     img1 = ax1.imshow(imentropy, cmap=plt.cm.jet)
#     ax1.set_title('Entropy')
#     ax1.axis('off')
#     plt.colorbar(img1, ax=ax1)
#     plt.show()
    return pixel_entropy #returns a matrix with entropy values for each pixel



def image_entropy(z):
    """calculate the entropy of an image"""
    img=data[:,:,z]
    histogram,_ = np.histogram(img,100)
    histogram_length = sum(histogram)

    samples_probability = [float(h) / histogram_length for h in histogram]
    image_entropy=-sum([p * math.log(p, 2) for p in samples_probability if p != 0])

    return image_entropy


#featurevector[9]= 3Dfeatures Ozekes and Osman (2010)

#featurevector[10]= edges

#featurevector[11]=uitgerektheid/compact

#featurevector[12]=convolution filters (filter banks)

#law filters (p296 ev)
#gabor filters
#eigenfilters


#featurevector[14]= Canny edge detection

# featurevector[10]= gradienten (slide 39 van feature selection pdf)

# featurevector[10]= Kadir and Brady algorithm (entropy)

# featurevector[10]= histogram distances (Manhattan distance, eucledian distance, max distance)
    # deel tekening op in kleine gebieden/ histogrammen en bekijk daar histogram distances ofzo
    

    
#featurevector[3]= 3D averaging (Keshani et al)
def averaging3D (x,y,z,windowrowvalue,data):
           
    # square windowrowvalue x windowrowvalue
    valdown = math.floor(windowrowvalue/2)
    valup = valdown+1
    
    windowDz = data[x-valdown:x+valup,y-valdown:y+valup,z]
    
    #reshape window into array to calculate mean (and variance)
    h,w,d = windowDz.shape()
    arrayD = np.reshape(windowDz, (h*w*d))
    
    Mz = arrayD.mean()
    
    # nodules will continue in preceeding/succeeding slices but bronchioles will not
    # assume: nodules have minimum length of 5 mm
    c = 5   # 5 mm
    T = Zsize   # thickness of slices
    q = c / T
    
    # mean of same window in preceding slices
    
    # mean of samen window in succeeding slices
    MzPlus = (1/z) * sum
    
    
    

#sklearn: image feature 2