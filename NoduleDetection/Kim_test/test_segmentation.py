import dicom
import pylab
import numpy as np
import numpy.ma as ma
import time
from scipy import ndimage

def getIntensityCounts(matrix, delta): #like a histogram with bin size=1
    counts = np.zeros(delta, dtype=np.int)
    for value in matrix:
        counts[value] += 1
    
    return counts

def binarizeImage(image, threshold):
    result = np.zeros(image.shape)
    for index,value in np.ndenumerate(image):
        result[index] = 0 if value < threshold else 1
        
    return result

def calcDistance(t, muHigh_i, muLow_i, i):
    if t <= i:
        return abs(t-muLow_i)
    else:
        return abs(t-muHigh_i)
    
#ds=dicom.read_file("../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000/000000.dcm")
ds=dicom.read_file("../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000/000007.dcm")
data=ds.pixel_array
#show image
#pylab.imshow(ds.pixel_array, cmap=pylab.gray())
#pylab.show()

#########################################################################################################
# STEP A
#########################################################################################################
# transform the pixel grey values to HU units
intercept = int(ds.RescaleIntercept) # found in dicom header at (0028,1052)
slope = int(ds.RescaleSlope) # found in dicom header at (0028,1053)
HU = data * slope - intercept
HU = HU // 10

# apply a mask to the image to exclude the pixels outside the thorax in the image
minI = HU.min()
maxI = HU.max()
thoraxMask = ma.masked_equal(HU, minI)
minI = thoraxMask.min() # find the new minimum inside mask region

if minI != 0: #shift intensities so that minI = 0
    thoraxMask -= minI
    maxI -= minI
    minI = 0

delta = maxI - minI + 1

print("grey levels: {} - {}".format(minI, maxI))

p = getIntensityCounts(thoraxMask, delta)

millis1=int(round(time.time()*1000))

Mlow=np.zeros(delta, dtype=np.int)
Mhigh=np.zeros(delta, dtype=np.int)
Tlow=np.zeros(delta, dtype=np.int)
Thigh=np.zeros(delta, dtype=np.int)
muLow=np.zeros(delta)
muHigh=np.zeros(delta)

sumT = p.sum()
sumM = 0
for i in range(delta):
    sumM += i*p[i]
    
for i in range(delta):  
    # step 1: calculate T and M for every grey value      
    for k in range(delta):
#         if k < i:
#             Mlow[i] += k*p[k]
#             Tlow[i] += p[k]
#         elif k > i:
#             Mhigh[i] += k*p[k]
#             Thigh[i] += p[k]
#         else: # k == i, calc both
#             Mlow[i] += k*p[k]
#             Tlow[i] += p[k]
#             Mhigh[i] += k*p[k]
#             Thigh[i] += p[k]

        if k >= i:
            Mhigh[i] += k*p[k]
            Thigh[i] += p[k]
             
    #assert sumT + p[i] == Tlow[i] + Thigh[i]
    Tlow[i] = sumT + p[i] - Thigh[i]
     
    #assert sumM + i*p[i] == Mlow[i] + Mhigh[i]
    Mlow[i] = sumM + i*p[i] - Mhigh[i]
    
    # step 2: calculate the mean values of both regions        
    muLow[i] = Mlow[i] / Tlow[i]
    muHigh[i] = Mhigh[i] / Thigh[i]

if False:
    pylab.subplot(221)
    pylab.title("$M_{low}$ (red) and $M_{high}$ (green)")
    pylab.xlabel("Grey Level")
    pylab.ylabel("M")
    pylab.plot(Mlow, 'r+')
    pylab.plot(Mhigh, 'g+')
    
    pylab.subplot(222)
    pylab.title("$T_{low}$ (red) and $T_{high}$ (green)")
    pylab.xlabel("Grey Level")
    pylab.ylabel("T")
    pylab.plot(Tlow, 'r+')
    pylab.plot(Thigh, 'g+')
    
    pylab.subplot(223)
    pylab.title("Histogram")
    pylab.xlabel("Grey Level")
    pylab.ylabel("Count")
    pylab.bar(np.arange(delta), p, 0.35)
    
    pylab.subplot(224)
    pylab.title("$\mu_{low}$ (red) and $\mu_{high}$ (green)")
    pylab.xlabel("Grey Level")
    pylab.ylabel("$\mu$")
    pylab.plot(muLow, 'r+')
    pylab.plot(muHigh, 'g+')
    pylab.show()

# print("Mhigh = {0}".format(Mhigh))
# print("Mlow = {0}".format(Mlow))
# print("Thigh = {0}".format(Thigh))
# print("Tlow = {0}".format(Tlow))

millis2=int(round(time.time()*1000))
  
# step 3: membership measurement
# step 4: determine cost function to find optimal threshold
C = np.zeros(delta)
prevC = 999999999
threshold = -1
for i in range(delta):
    for t in range(delta):
        d = calcDistance(t, muHigh[i], muLow[i], i)
        m = 1 / (1 + (d / (maxI - 1)))
        C[i] += (m * (1 - m))**2 #t in [minI, maxI-1]
    
    #print("C[%d] = %d" % (i, C[i]))     
    if C[i] < prevC:
        threshold=i # minimal cost function determines grey level for threshold
        prevC = C[i]

#threshold = 153 
print("Optimal threshold: %d" % threshold)
 
millis3=int(round(time.time()*1000))

# step 5: binarization image        
result = binarizeImage(HU, threshold)

nonLungMask = ma.masked_greater(HU, threshold)
combinedMask = ma.mask_or(ma.getmask(thoraxMask), ma.getmask(nonLungMask))
combinedMask = ma.array(HU, mask=combinedMask) #apply on matrix

pylab.subplot(1, 2, 1)
pylab.imshow(thoraxMask, cmap=pylab.gray())

pylab.subplot(1, 2, 2)
pylab.imshow(combinedMask, cmap=pylab.gray())
pylab.show()

print("Step A1-2: %dms" % (millis2-millis1))
print("Step A3-4: %dms" % (millis3-millis2))

################################################################################
# STEP B
################################################################################
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

