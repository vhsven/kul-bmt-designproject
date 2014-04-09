import pylab
from matplotlib.widgets import Slider
import numpy as np
import numpy.ma as ma
from scipy import ndimage
from skimage.morphology import reconstruction
from DicomFolderReader import DicomFolderReader 


def getHistogram(img, minI, maxI):
    binEdges = np.arange(minI, maxI + BIN_SIZE, BIN_SIZE)
    p, _ = np.histogram(img, binEdges)
    return p
    
def calcThresholds(mySlice, low, high):
    HU = dfr.getSlicePixelsRescaled(mySlice)
    #minI = HU.min()
    #maxI = HU.max()
    #print("Slice {}: rescaled grey levels {} -- {}".format(mySlice, minI, maxI))
    #p = getHistogram(HU, minI, maxI)
    
    masked = ma.masked_inside(HU, low, high)
    
    #opening
    newmask = ndimage.binary_opening(masked.mask, structure=np.ones((9,9)), iterations=1).astype(np.int)
    #newmask = ndimage.binary_dilation(newmask, structure=np.ones((9,9)), iterations=1).astype(np.int)
    #masked.mask = newmask 
    #reconstruction = ndimage.binary_propagation(eroded_square, mask=square)

    seed = np.copy(newmask)
    seed[0:-1, 0:-1] = newmask.max()
    newmask = reconstruction(seed, newmask, method='erosion')

    minI = masked.min() # find the new minimum inside mask region
    maxI = masked.max()
    p = getHistogram(masked, minI, maxI)
    barX = np.arange(minI, maxI, BIN_SIZE)
    
    return masked, p, barX, newmask
    
BIN_SIZE = 10
LOW_INIT = 1600
HIGH_INIT = 4000

#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
dfr = DicomFolderReader(myPath)

#pylab.ion() #interactive mode: don't stop on show()
#pylab.figure.max_num_figures = 50
fig, ax = pylab.subplots()
pylab.subplots_adjust(bottom=0.25)

#masked, p, barX, mask2 = calcThresholds(0, LOW_INIT, HIGH_INIT)
sp1 = pylab.subplot(131)
sp2 = pylab.subplot(132)
sp3 = pylab.subplot(133)
    
#axes: left, bottom, width, height
sSlider = Slider(pylab.axes([0.1, 0.15, 0.8, 0.03]), 'Slice', 0, dfr.getNbSlices()-1, 0, valfmt='%1.0f')
minSlider = Slider(pylab.axes([0.1, 0.10, 0.8, 0.03]), 'MinI', 0, 4000, LOW_INIT, valfmt='%1.0f')
maxSlider = Slider(pylab.axes([0.1, 0.05, 0.8, 0.03]), 'MaxI', 0, 4000, HIGH_INIT, valfmt='%1.0f')

def update(val):
    mySlice = int(sSlider.val)
    low = int(minSlider.val)
    high = int(maxSlider.val)
    
    masked, p, barX, mask2 = calcThresholds(mySlice, low, high)
    sp1.clear()
    sp2.clear()
    sp3.clear()
    
    sp1.imshow(masked.mask, cmap=pylab.gray())
    sp2.bar(barX, p, 0.35)
    sp3.imshow(mask2, cmap=pylab.gray())
    
    fig.canvas.draw_idle()

sSlider.on_changed(update)
minSlider.on_changed(update)
maxSlider.on_changed(update)

update(0)
pylab.show()

#Idee 1:
# - segmenteer thorax wall: 1600+
# - floodfill/imfill alles erbuiten
# => alleen longen zijn nog wit
# - optimaliseer met erosion/dilation

#Idee 2:
# - trek 2 verticale lijnen op x={25%, 75%}
# - zoek eerste/laatste witte pixel na/voor zwarte regio
# => y-coordinaten voor bounding box 