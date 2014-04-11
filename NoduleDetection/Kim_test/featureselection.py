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
import scipy as sp
import scipy.ndimage as nd
from scipy.ndimage.filters import generic_gradient_magnitude, sobel




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
#featurevector[2]= greyvalue + related features in window
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
    h,w,d=windowD.shape
    arrayD = np.reshape(windowD, (h*w*d))
    
    # mean and variance
    M=arrayD.mean()
    V=arrayD.var()
    
    rangex = range(w)
    rangey = range(h)
    rangez = range(d)


    #calculate projections along the x and y axes
    zp = np.sum(windowD,axis=2)
    yp = np.sum(windowD,axis=1)
    xp = np.sum(windowD,axis=0)

    #centroid
    cx = np.sum(rangex*xp)/np.sum(xp)
    cy = np.sum(rangey*yp)/np.sum(yp)
    cz = np.sum(rangez*zp)/np.sum(zp)

    #standard deviation
    x2 = (rangex-cx)**2
    y2 = (rangey-cy)**2
    z2 = (rangez-cz)**2

    sx = np.sqrt( np.sum(x2*xp)/np.sum(xp) )
    sy = np.sqrt( np.sum(y2*yp)/np.sum(yp) )
    sz = np.sqrt( np.sum(z2*zp)/np.sum(zp) )

    #skewness
    x3 = (rangex-cx)**3
    y3 = (rangey-cy)**3
    z3 = (rangez-cz)**3

    skx = np.sum(xp*x3)/(np.sum(xp) * sx**3)
    sky = np.sum(yp*y3)/(np.sum(yp) * sy**3)
    skz = np.sum(zp*z3)/(np.sum(zp) * sz**3)

    #Kurtosis
    x4 = (rangex-cx)**4
    y4 = (rangey-cy)**4
    z4 = (rangez-cz)**4
    kx = np.sum(xp*x4)/(np.sum(xp) * sx**4)
    ky = np.sum(yp*y4)/(np.sum(yp) * sy**4)
    kz = np.sum(zp*z4)/(np.sum(zp) * sz**4)
    
    #autocorrelation
    #result = np.correlate(arrayD, arrayD, mode='full')
    #autocorr=result[result.size/2:]
    
    # maximum and minimum greyvalue of pixels in window
    Max_greyvalue = arrayD.max()
    Min_greyvalue = arrayD.min()
    
    # difference between greyvalue pixel and max/min grey value
    maxdiff = data[x,y,z] - Max_greyvalue
    mindiff = data[x,y,z] - Min_greyvalue
    
    maxplus = data[x,y,z] + Max_greyvalue
    minplus = data[x,y,z] + Min_greyvalue
    
    maxdiv = data[x,y,z]/Max_greyvalue
    mindiv = data[x,y,z]/Min_greyvalue
    
    maxmindiff = Max_greyvalue - Min_greyvalue
    
    # COUNT VALUE PIXEL/MAX/MIN IN WINDOW
    counter = collections.Counter(arrayD)
    freq_pixelvalue = counter[data[x,y,z]] # prevalence of pixelvalue in window
    
    freq_max = counter[Max_greyvalue]
    freq_min = counter[Min_greyvalue]
    
       
    
    return greyvalue,M,V,cx,cy,cz,sx,sy,sz,skx,sky,skz,kx,ky,kz,Max_greyvalue,Min_greyvalue,maxdiff,mindiff,maxdiv,minplus,maxplus, mindiv,maxmindiff,freq_pixelvalue,freq_max,freq_min

def neighbours(x,y,z,data):
    # top - bottom neighbours
    Ptop = data[x,y-1,z]
    Pbottom = data[x,y+1,z]
    
    Ptbmin = Ptop - Pbottom
    Ptbdiv = Ptop/Pbottom
    Ptbplus = Ptop + Pbottom
    
    Ppixeltopmin = data[x,y,z] - Ptop
    Ppixelbottommin = data[x,y,z] - Pbottom
    
    Ppixeltopplus = data[x,y,z] + Ptop
    Ppixelbottomplus = data[x,y,z] + Pbottom
    
    Ppixeltopdiv = data[x,y,z] / Ptop
    Ppixelbottomdiv = data[x,y,z] / Pbottom
    
        
    # left - right neighbours
    PL = data[x-1,y,z]
    PR = data[x+1,y,z]
    
    PLRmin = PL - PR
    PLRdiv = PL/PR
    PLRplus = PL + PR
    
    PpixelLmin = data[x,y,z] - PL
    PpixelRmin = data[x,y,z] - PR
    
    PpixelLplus = data[x,y,z] + PL
    PpixelRplus = data[x,y,z] + PR
    
    PpixelLdiv = data[x,y,z] / PL
    PpixelRdiv = data[x,y,z] / PR
    
        
    # front - back neighbours
    Pf = data[x,y,z-1]
    Pb = data[x,y,z+1]
    Pfbmin = Pf - Pb
    Pfbdiv = Pf/Pb
    Pfbplus = Pf + Pb
    
    Ppixelfmin = data[x,y,z] - Pf
    Ppixelbmin = data[x,y,z] - Pb
    
    Ppixelfplus = data[x,y,z] + Pf
    Ppixelbplus = data[x,y,z] + Pb
    
    Ppixelfdiv = data[x,y,z] / Pf
    Ppixelbdiv = data[x,y,z] / Pb
    
    return Ptop, Pbottom, Ptbmin, Ptbdiv, Ptbplus, Ppixeltopmin, Ppixelbottommin, Ppixeltopplus, Ppixelbottomplus, Ppixeltopdiv, Ppixelbottomdiv, PL, PR, PLRmin, PLRdiv, PLRplus, PpixelLmin, PpixelRmin, PpixelLplus, PpixelRplus, PpixelLdiv, PpixelRdiv, Pf, Pb, Pfbmin, Pfbdiv, Pfbplus, Ppixelfmin, Ppixelbmin, Ppixelfplus, Ppixelbplus, Ppixelfdiv, Ppixelbdiv
    
    
   
############################################################
#featurevector[3]= prevalence of that grey value
############################################################
def greyvaluefrequency(x,y,z,data):
    m,n,d = data.shape
    mydata = np.reshape(data, (m*n*d))
    #import collections
    counter = collections.Counter(mydata)
    freqvalue = counter[data[x,y,z]] # prevalence of pixelvalue in image
    
    # prevalence maximum and minimum of pixels in image
    # max and min
    Max_image = mydata.max()
    Min_image = mydata.min()
    
    # prevalence max and min
    freqmax = counter[Max_image]
    freqmin = counter[Min_image]
    
    # compare (prevalence of) pixelvalue to min and max (prevalence)
    comfreq_max = freqvalue/freqmax
    comfreq_min = freqvalue/freqmin
    
    rel_max = data[x,y,z]/Max_image
    rel_min = data[x,y,z]/Min_image
    
    
    return freqvalue, comfreq_max, comfreq_min, rel_max, rel_min


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
    
    
    return gradUD, gradmeanUD, divUD, divmeanUD


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


############################################################
# feature[8]= 3D averaging (Keshani et al)
############################################################

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
    Q = c / T
       
    # mean of same window in preceding slices
    windowDzmin = data[x-valdown:x+valup,y-valdown:y+valup,z-Q:z-1]
    
    h,w,d = windowDzmin.shape
    arrayDmin = windowDzmin.reshape(h*w*d) # make array of 3D matrix
    matrixDmin = arrayDmin.reshape(d, h*w) # make matrix with every row the values of the window per slice
    trans1 = matrixDmin.transpose() # switch rows and columns
    row,col = trans1.shape
    
    S1 = sum(trans1)/row
    Mpre = S1.mean()
    
    
    # mean of same window in succeeding slices
    windowDzplus = data[x-valdown:x+valup,y-valdown:y+valup,z+1:z+Q]
       
    h,w,d = windowDzplus.shape
    arrayDplus = windowDzplus.reshape(h*w*d) # make array of 3D matrix
    matrixDplus = arrayDplus.reshape(d, h*w) # make matrix with every row the values of the window per slice
    trans2 = matrixDplus.transpose() # switch rows and columns
    row,col = trans2.shape
    
    S2 = sum(trans2)/row
    Mplus = S2.mean()
    
    return Mz, Mpre, Mplus # REMARK: Mz is common mean (also in previous function)


############################################################
# feature[9]= gradients: sobel
############################################################

def gradients(x,y,z,data):
    #import scipy
    #from scipy import ndimage
    #from scipy.ndimage.filters import generic_gradient_magnitude, sobel
                       
#     dx = ndimage.sobel(data, 0)  # x derivative
#     dy = ndimage.sobel(data, 1)  # y derivative
#     dz = ndimage.sobel(data, 2)  # z derivative
    
    mag = generic_gradient_magnitude(data, sobel)
        
    return mag

############################################################
# feature[10]= blob detection with laplacian of gaussian
############################################################
def blobdetection(x,y,z,data):
    # REMARK: DATA SHOULD BE 2D
    #data = ds.pixel_array
    LoG = nd.gaussian_laplace(data, 1.8) # scalar: standard deviations of the Gaussian filter
    # sigma empirisch vastgesteld op 1.9/ 2/ 2.1
    aLoG = abs(LoG)
    output = np.copy(data)
    output[aLoG > aLoG.max()-200] = 1
    #pylab.imshow(output, cmap=pylab.gray())
    #pylab.show()
    return output

############################################################  
#featurevector[10]=haar features
############################################################



    
