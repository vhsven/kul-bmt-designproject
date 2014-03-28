'''
Created on 27-mrt.-2014

@author: Eigenaar
'''
import dicom
import pylab
import numpy
import time

# get image slice
ds=dicom.read_file("../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000/000000.dcm")
#print(ds)
# get pixel values
data=ds.pixel_array
#print(data)
#show image
#pylab.imshow(ds.pixel_array, cmap=pylab.gray())
#pylab.show()

#########################################################################################################
# STEP A
######################################################################################################### 
# apply a mask to the image to exclude the pixels outside the thorax in the image
#transform the pixel grey values to HU units: HU = pixel_value*slope - intercept
intercept = ds.RescaleIntercept # found in dicom header at (0028,1052)
slope = ds.RescaleSlope # found in dicom header at (0028,1053)
HU=data*slope - intercept
HU = HU / 3
maxI=int(HU.max())
minI=int(HU.min())
# kom max waarde 3195 uit en min 0??? long zou normaal -500 moeten zijn

assert minI==0
datavector=HU.reshape(512*512,1)
print(datavector.shape)
(p, _, _)=pylab.hist(datavector,maxI)
pylab.show()
print(p)

# get rid of grey values with zero intensities
myset=[]
for i in range(0,len(p)):
    if p[i] !=0:
        myset.append(i)
        
print(myset)

## step 1: calculate T and M for every grey value
millis1=int(round(time.time()*1000))
Mhigh=numpy.zeros(maxI - minI)
Thigh=numpy.zeros(maxI - minI)
Mlow=numpy.zeros(maxI - minI)
Tlow=numpy.zeros(maxI - minI)
for i in myset:
    Mhigh[i]=0
    Mlow[i]=0
    
    Thigh[i]=0
    Tlow[i]=0
        
    for k in myset:
        if k <= i:
            Mlow[i]=k*p[k]+Mlow[i]
            Tlow[i]=p[k]+Tlow[i]
            
        if k >= i:
            Mhigh[i]=k*p[k]+Mhigh[i]
            Thigh[i]=p[k]+Thigh[i]
        
    #for k in range(i,maxI):
            
    #for k in range(minI,i):

print("Mhigh = %s" % Mhigh )  
print("Thigh = %s" % Thigh )
print("Mlow = %s" % Mlow)
print("Tlow = %s" % Tlow )
# print("Mhigh = {0}".format(Mhigh))
# print("Mlow = {0}".format(Mlow))
# print("Thigh = {0}".format(Thigh))
# print("Tlow = {0}".format(Tlow))

millis2=int(round(time.time()*1000))
# step 2: calculate the mean values of both regions

mulow={}
muhigh={}

for i in range(minI, maxI):
    if Tlow[i] == 0:
        mulow[i]=0
    else:
        mulow[i]=Mlow[i]/Tlow[i]
    
    if Thigh[i] == 0:
        muhigh[i]=0
    else:
        muhigh[i]=Mhigh[i]/Thigh[i]

millis3=int(round(time.time()*1000))    
# step 3: membership measurement
def distance(t, muHigh_i, muLow_i, i):
    if t <= i:
        return abs(t-muLow_i)
    else:
        return abs(t-muHigh_i)

delta = maxI - minI
print("delta: {0}".format(delta))
member = numpy.zeros((maxI - minI)**2).reshape(maxI - minI, maxI - minI)

millis4=int(round(time.time()*1000))
# step 4: determine cost function to find optimal threshold Io
# C = {}
# for i in range(minI, maxI):
#     for t in range(minI, maxI):
#         member[i,t]= 1/(1 + distance(t, muhigh[i], mulow[i], i) / (maxI - 1))
#    
#     C[i] = 0
#     for t in range(minI, maxI-1):
#         C[i] += (member[i,t] * (1 - member[i,t]))**2
#         
#         if i>1 and C[i] < C[i-1]:
#             Io=i # minimal cost function determines grey level for threshold
# print(Io)
# 
# millis5=int(round(time.time()*1000))
# 
# # step 5: binarization image
# n,m=HU.shape
# for i in range(0,n-1):
#     for j in range(0,m-1):
#         if HU[i,j] < Io:
#             HU[i,j]=0
#         else:
#             HU[i,j]=1
# 
# pylab.imshow(HU, cmap=pylab.gray())
# pylab.show()

print(millis2-millis1)
print(millis3-millis2)
print(millis4-millis3)
#print(millis5-millis4)
################################################################################
# STEP B
################################################################################
# from scipy import ndimage
# import numpy as np
# import pylab
# square = np.zeros((32, 32))
# square[10:-10, 10:-10] = 1
# np.random.seed(2)
# x, y = (32*np.random.random((2, 20))).astype(np.int)
# square[x, y] = 1
# open_square = ndimage.binary_opening(square)
# pylab.imshow(open_square, cmap=pylab.gray())
# pylab.show()
# eroded_square = ndimage.binary_erosion(square)
# pylab.imshow(eroded_square, cmap=pylab.gray())
# pylab.show()
# reconstruction = ndimage.binary_propagation(eroded_square, mask=square)
# pylab.imshow(reconstruction, cmap=pylab.gray())
# pylab.show()

