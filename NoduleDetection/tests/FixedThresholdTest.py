import pylab
from matplotlib.widgets import Slider
import numpy as np
import numpy.ma as ma
from skimage.morphology import reconstruction, binary_opening, binary_erosion
from DicomFolderReader import DicomFolderReader 

BIN_SIZE = 10
DEFAULT_THRESHOLD = 1600

def getHistogram(img, minI, maxI):
    binEdges = np.arange(minI, maxI + BIN_SIZE, BIN_SIZE)
    p, _ = np.histogram(img, binEdges)
    barX = np.arange(minI, maxI, BIN_SIZE)
    return p, barX

def processVolume(threshold):
    print("Generating Volume Data")
    data = dfr.getVolumeData()
    print("Applying threshold")
    masked = ma.masked_greater(data, threshold)
    print("Performing binary opening")
    newmask = binary_opening(masked.mask, selem=np.ones((9,9,9)))
    seed = np.copy(newmask)
    seed[1:-1, 1:-1, 1:-1] = newmask.max()
    print("Performing reconstruction")
    newmask = reconstruction(seed, newmask, method='erosion').astype(np.int)
    #newmask = binary_erosion(newmask, selem=np.ones((29,29,29))).astype(np.int)
    masked2 = ma.array(data, mask=np.logical_not(newmask))
    
    return masked, masked2
    
def processSlice(mySlice, threshold):
    HU = dfr.getSlicePixelsRescaled(mySlice)
    
    # select pixels in thorax wall
    masked = ma.masked_greater(HU, threshold)
    
    # opening to remove small structures
    newmask = binary_opening(masked.mask, selem=np.ones((9,9)))

    # reconstruction to fill lungs (figure out how this works) 
    seed = np.copy(newmask)
    seed[1:-1, 1:-1] = newmask.max()
    newmask = reconstruction(seed, newmask, method='erosion').astype(np.int)
    
    # erode thorax walls slightly (no nodules in there)
    newmask = binary_erosion(newmask, selem=np.ones((29,29))).astype(np.int)
    
    masked2 = ma.array(HU, mask=np.logical_not(newmask))
    
    return masked, masked2

#myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
dfr = DicomFolderReader(myPath)

maskedV, maskedV2 = processVolume(DEFAULT_THRESHOLD)

fig, ax = pylab.subplots()
pylab.subplots_adjust(bottom=0.20)

sp1 = pylab.subplot(121)
sp2 = pylab.subplot(122)
    
#axes: left, bottom, width, height
sSlider = Slider(pylab.axes([0.1, 0.10, 0.8, 0.03]), 'Slice', 0, dfr.getNbSlices()-1, 0, valfmt='%1.0f')
tSlider = Slider(pylab.axes([0.1, 0.05, 0.8, 0.03]), 'Threshold', 0, 4000, DEFAULT_THRESHOLD, valfmt='%1.0f')

def update(val):
    mySlice = int(sSlider.val)
    threshold = int(tSlider.val)
    
    #masked, masked2 = processSlice(mySlice, threshold)
    masked, masked2 = maskedV[:,:,mySlice], maskedV2[:,:,mySlice]
    
    sp1.clear()
    sp2.clear()
    
    sp1.imshow(masked, cmap=pylab.gray())
    sp2.imshow(masked2, cmap=pylab.gray())
    
    fig.canvas.draw_idle()

sSlider.on_changed(update)
tSlider.on_changed(update)

update(0)
pylab.show()