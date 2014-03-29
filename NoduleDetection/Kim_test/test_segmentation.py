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
# transform the pixel grey values to HU units: HU = pixel_value*slope - intercept
intercept = int(ds.RescaleIntercept) # found in dicom header at (0028,1052)
slope = int(ds.RescaleSlope) # found in dicom header at (0028,1053)
HU = data * slope - intercept
#HU = HU // 10
maxI = HU.max()
minI = HU.min()
delta = maxI - minI + 1
# kom max waarde 3195 uit en min 0??? long zou normaal -500 moeten zijn

assert minI==0

def getIntensityCounts(matrix):
    counts = numpy.zeros(maxI+1, dtype=numpy.int)
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            counts[matrix[i,j]] += 1
    return counts

p = getIntensityCounts(HU)
print(p)

# create list of all intensities that occur in the image
# better than looping over all minI -- maxI
myset=set(HU.flatten())
        
print(myset)

millis1=int(round(time.time()*1000))

Mlow=numpy.zeros(delta, dtype=numpy.int)
Mhigh=numpy.zeros(delta, dtype=numpy.int)
Tlow=numpy.zeros(delta, dtype=numpy.int)
Thigh=numpy.zeros(delta, dtype=numpy.int)
muLow=numpy.zeros(delta)
muHigh=numpy.zeros(delta)
for i in myset:  
    # step 1: calculate T and M for every grey value      
    for k in myset:
        if k < i:
            Mlow[i] += k*p[k]
            Tlow[i] += p[k]
        elif k > i:
            Mhigh[i] += k*p[k]
            Thigh[i] += p[k]
        else: # k == i, calc both
            Mlow[i] += k*p[k]
            Tlow[i] += p[k]
            Mhigh[i] += k*p[k]
            Thigh[i] += p[k]

    # step 2: calculate the mean values of both regions       
    muLow[i] = Mlow[i] / Tlow[i]
    muHigh[i] = Mhigh[i] / Thigh[i]

print("Mlow = %s" % Mlow)
print("Mhigh = %s" % Mhigh )  
print("Tlow = %s" % Tlow )
print("Thigh = %s" % Thigh )
print("mulow = %s" % muLow )
print("muHigh = %s" % muHigh )
# print("Mhigh = {0}".format(Mhigh))
# print("Mlow = {0}".format(Mlow))
# print("Thigh = {0}".format(Thigh))
# print("Tlow = {0}".format(Tlow))

millis2=int(round(time.time()*1000))
  
# step 3: membership measurement
def calcDistance(t, muHigh_i, muLow_i, i):
    if t <= i:
        return abs(t-muLow_i)
    else:
        return abs(t-muHigh_i)

# step 4: determine cost function to find optimal threshold Io
C = numpy.zeros(delta)
prevC = 999999999
threshold = -1
for i in myset:
    for t in myset:
        d = calcDistance(t, muHigh[i], muLow[i], i)
        m = 1 / (1 + (d / (maxI - 1)))
        C[i] += (m * (1 - m))**2 #t in [minI, maxI-1]
    
    #print("C[%d] = %d" % (i, C[i]))     
    if C[i] < prevC:
        threshold=i # minimal cost function determines grey level for threshold
            
    prevC = C[i]

#threshold = 210   
print("Optimal threshold: %d" % threshold)
 
millis3=int(round(time.time()*1000))

# step 5: binarization image
for index,value in numpy.ndenumerate(HU):
    HU[index] = 0 if value < threshold else 1

print("Step A1-2: %dms" % (millis2-millis1))
print("Step A3-4: %dms" % (millis3-millis2))

pylab.imshow(HU, cmap=pylab.gray())
pylab.show()

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

