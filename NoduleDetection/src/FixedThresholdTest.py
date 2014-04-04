import pylab
from matplotlib.widgets import Slider
import numpy as np
import numpy.ma as ma
from DicomFolderReader import DicomFolderReader 


def getHistogram(img, minI, maxI):
    binEdges = np.arange(minI, maxI + BIN_SIZE, BIN_SIZE)
    bins = len(binEdges) - 1
    p, _ = np.histogram(img, binEdges)
    return p
    
def calcThresholds(mySlice):
    HU = dfr.getSlicePixelsRescaled(mySlice)
    minI = HU.min()
    maxI = HU.max()
    print("Slice {}: rescaled grey levels {} -- {}".format(mySlice, minI, maxI))
    #p = getHistogram(HU, minI, maxI)
    
    #STEP 1: exclude pixels having minimum value (outside thorax)
    thoraxMask = ma.masked_equal(HU, minI)
    minI = thoraxMask.min() # find the new minimum inside mask region
#     p = getHistogram(thoraxMask, minI, maxI)
#     
#     pylab.subplot(121)
#     pylab.title("Thorax - Slice {}".format(mySlice))
#     pylab.imshow(thoraxMask, cmap=pylab.gray())
#     pylab.subplot(122)
#     pylab.title("Histogram")
#     pylab.bar(np.arange(minI, maxI, BIN_SIZE), p, 0.35)
#     pylab.show()
    
    #STEP 2: only keep dark values (air in lungs)
    darkMask = ma.masked_greater(thoraxMask, 1500)
    maxI = darkMask.max()
    p = getHistogram(thoraxMask, minI, maxI)
    barX = np.arange(minI, maxI, BIN_SIZE)
    
    return darkMask, p, barX
    
BIN_SIZE = 8
myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
dfr = DicomFolderReader(myPath)

#pylab.ion() #interactive mode: don't stop on show()
#pylab.figure.max_num_figures = 50
fig, ax = pylab.subplots()
pylab.subplots_adjust(bottom=0.25)

darkMask, p, barX = calcThresholds(0)
sp1 = pylab.subplot(121)
pylab.title("Dark Regions - Slice {}".format(0))
pylab.imshow(darkMask, cmap=pylab.gray())
sp2 = pylab.subplot(122)
pylab.title("Histogram")
pylab.bar(barX, p, 0.35)
    
axSlider = pylab.axes([0.25, 0.1, 0.65, 0.03])
sSlider = Slider(axSlider, 'Slice', 0, dfr.getNbSlices()-1, 0)

def update(val):
    mySlice = int(sSlider.val)
    
    darkMask, p, barX = calcThresholds(mySlice)
    sp1.imshow(darkMask, cmap=pylab.gray())
    sp2.bar(barX, p, 0.35)
    
    fig.canvas.draw_idle()
sSlider.on_changed(update)

pylab.show()