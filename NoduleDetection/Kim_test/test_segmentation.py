import pylab
import numpy as np
import numpy.ma as ma
import time
from scipy import ndimage
from DicomFolderReader import DicomFolderReader 

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

SHOW_PLOTS = True
BIN_SIZE = 16

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
dfr = DicomFolderReader(myPath)
ds = dfr.Slices[50]
data = ds.pixel_array
minI = data.min()
maxI = data.max()
print("raw grey levels: {} - {}".format(minI, maxI))

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

# apply a mask to the image to exclude the pixels outside the thorax in the image
minI = HU.min()
maxI = HU.max()
print("rescaled grey levels: {} - {}".format(minI, maxI))
thoraxMask = ma.masked_equal(HU, minI)
minI = thoraxMask.min() # find the new minimum inside mask region

if minI != 0: #shift intensities so that minI = 0
    thoraxMask -= minI
    maxI -= minI
    minI = 0

delta = maxI - minI + 1

print("masked/shifted grey levels: {} - {}".format(minI, maxI))

peak1Mask = ma.masked_outside(thoraxMask, BIN_SIZE*0 ,  BIN_SIZE*5)  #  0 -   80
peak2Mask = ma.masked_outside(thoraxMask, BIN_SIZE*5 ,  BIN_SIZE*15) # 80 -  240
peak3Mask = ma.masked_outside(thoraxMask, BIN_SIZE*50 , BIN_SIZE*60) #800 -  960
peak4Mask = ma.masked_outside(thoraxMask, BIN_SIZE*60 , BIN_SIZE*75) #960 - 1200
fixedMask = ma.masked_greater(HU, 1500)

pylab.subplot(231)
pylab.title("Peak 1")
pylab.imshow(peak1Mask, cmap=pylab.gray())
pylab.subplot(232)
pylab.title("Peak 2")
pylab.imshow(peak2Mask, cmap=pylab.gray())
pylab.subplot(234)
pylab.title("Peak 3")
pylab.imshow(peak3Mask, cmap=pylab.gray())
pylab.subplot(235)
pylab.title("Peak 4")
pylab.imshow(peak4Mask, cmap=pylab.gray())
pylab.subplot(236)
pylab.title("> 1500")
pylab.imshow(fixedMask, cmap=pylab.gray())
pylab.show()

#exit(0)

binEdges = np.arange(minI, maxI + BIN_SIZE, BIN_SIZE)
bins = len(binEdges) - 1
p, _ = np.histogram(thoraxMask, binEdges)
millis1=int(round(time.time()*1000))

Mlow=np.zeros(bins, dtype=np.int)
Mhigh=np.zeros(bins, dtype=np.int)
Tlow=np.zeros(bins, dtype=np.int)
Thigh=np.zeros(bins, dtype=np.int)
muLow=np.zeros(bins)
muHigh=np.zeros(bins)

sumT = sum(p)
sumM = sum(range(bins) * p)
    
for i in range(bins):  
    # step 1: calculate T and M for every grey value      
    k = range(i, bins)
    Mhigh[i] = sum(k * p[i:])
    Thigh[i] = sum(p[i:])
    
    #assert sumT + p[i] == Tlow[i] + Thigh[i]
    Tlow[i] = sumT + p[i] - Thigh[i]
    
    #assert sumM + i*p[i] == Mlow[i] + Mhigh[i]
    Mlow[i] = sumM + i*p[i] - Mhigh[i]
    
    # step 2: calculate the mean values of both regions        
    muLow[i] = Mlow[i] / Tlow[i] #TODO check division by zero
    muHigh[i] = Mhigh[i] / Thigh[i]

if SHOW_PLOTS:
    pylab.subplot(231)
    pylab.title("$M_{low}$ (red) and $M_{high}$ (green)")
    pylab.xlabel("Grey Level")
    pylab.ylabel("M")
    pylab.plot(Mlow, 'r+')
    pylab.plot(Mhigh, 'g+')
    
    pylab.subplot(232)
    pylab.title("$T_{low}$ (red) and $T_{high}$ (green)")
    pylab.xlabel("Grey Level")
    pylab.ylabel("T")
    pylab.plot(Tlow, 'r+')
    pylab.plot(Thigh, 'g+')
    
    pylab.subplot(234)
    pylab.title("Histogram")
    pylab.xlabel("Grey Level")
    pylab.ylabel("Count")
    pylab.bar(np.arange(bins), p, 0.35)
    
    pylab.subplot(235)
    pylab.title("$\mu_{low}$ (red) and $\mu_{high}$ (green)")
    pylab.xlabel("Grey Level")
    pylab.ylabel("$\mu$")
    pylab.plot(muLow, 'r+')
    pylab.plot(muHigh, 'g+')

# print("Mhigh = {0}".format(Mhigh))
# print("Mlow = {0}".format(Mlow))
# print("Thigh = {0}".format(Thigh))
# print("Tlow = {0}".format(Tlow))

millis2=int(round(time.time()*1000))
  
# step 3: membership measurement
# step 4: determine cost function to find optimal threshold
C = np.zeros(bins)
Member = np.zeros(bins ** 2).reshape(bins, bins)
for i in range(bins):
    for t in range(bins):
        d = calcDistance(t, muHigh[i], muLow[i], i)
        m = 1 / (1 + (d / (maxI - 1)))
        Member[t][i] = m
        C[i] += (m * (1 - m))**2 #t in [minI, maxI-1]

threshold = C.argmin() # minimal cost function index determines grey level for threshold

if SHOW_PLOTS:
    pylab.subplot(233)
    pylab.title("Membership Measurement")
    pylab.xlabel("Grey Level t")
    pylab.ylabel("Grey level i")
    pylab.imshow(Member, origin='lower')
    pylab.colorbar()
    
    pylab.subplot(236)
    pylab.title("Cost Function")
    pylab.xlabel("Grey Level")
    pylab.ylabel("C")
    pylab.plot(C, 'k+')
    pylab.show()

#threshold = 150
threshold *= BIN_SIZE #convert bin back to intensity
print("Optimal threshold: %d" % threshold)
 
millis3=int(round(time.time()*1000))

# step 5: binarization image
nonLungMask = ma.masked_greater(HU, threshold) #threshold is in non-shifted waarden???
combinedMask = ma.mask_or(ma.getmask(thoraxMask), ma.getmask(nonLungMask))
combinedMask = ma.array(HU, mask=combinedMask) #apply on matrix
#combinedMask = ma.masked_greater(thoraxMask, threshold)

print("Step A1-2: %dms" % (millis2-millis1))
print("Step A3-4: %dms" % (millis3-millis2))

pylab.subplot(1, 2, 1)
pylab.imshow(thoraxMask, cmap=pylab.gray())

pylab.subplot(1, 2, 2)
pylab.imshow(combinedMask, cmap=pylab.gray())
pylab.show()

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

