import dicom
import pylab
import numpy as np
import time

def getIntensityCounts(matrix): #like a histogram with bin size=1
    counts = np.zeros(maxI+1, dtype=np.int)
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            counts[matrix[i,j]] += 1
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
ds=dicom.read_file("../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000/000000.dcm")
data=ds.pixel_array
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
HU = HU // 10
minI = HU.min()

if minI != 0: #shift intensities so that minI = 0
    HU -= minI
    minI = 0

maxI = HU.max()
delta = maxI - minI + 1

print("grey levels: {} - {}".format(minI, maxI))

mask = binarizeImage(HU, 1) #ignore all darkest pixels (outside of thorax)
#pylab.imshow(mask, cmap=pylab.gray())
#pylab.show()

p = getIntensityCounts(HU)
print(p)

# create list of all intensities that occur in the image
# better than looping over all minI -- maxI
myset=set(HU.flatten())
        
print(myset)

millis1=int(round(time.time()*1000))

Mlow=np.zeros(delta, dtype=np.int)
Mhigh=np.zeros(delta, dtype=np.int)
Tlow=np.zeros(delta, dtype=np.int)
Thigh=np.zeros(delta, dtype=np.int)
muLow=np.zeros(delta)
muHigh=np.zeros(delta)

sumT = p.sum()
sumM = 0
for i in myset:
    sumM += i*p[i]
    
for i in myset:  
    # step 1: calculate T and M for every grey value      
    for k in myset:
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
# step 4: determine cost function to find optimal threshold Io
C = np.zeros(delta)
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
result = binarizeImage(HU, threshold)

print("Step A1-2: %dms" % (millis2-millis1))
print("Step A3-4: %dms" % (millis3-millis2))

pylab.imshow(result, cmap=pylab.gray())
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

