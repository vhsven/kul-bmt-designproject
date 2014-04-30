import pylab
import numpy as np
import scipy.ndimage as nd
from matplotlib.widgets import Slider
from DicomFolderReader import DicomFolderReader
from XmlAnnotationReader import XmlAnnotationReader
from scipy.ndimage.filters import generic_gradient_magnitude, sobel
from Preprocessor import Preprocessor

setID = int(raw_input("Load dataset #: "))
myPath = DicomFolderReader.findPath("../data/LIDC-IDRI", setID)
dfr = DicomFolderReader(myPath)
cc = dfr.getCoordinateConverter()
reader = XmlAnnotationReader(myPath, cc)
for c, r in reader.getNodulePositions():
    print c, r
vh,vw,vd = dfr.getVoxelShape()
voxelShape = np.array([vh,vw,vd])
mask3D = Preprocessor.loadThresholdMask(setID)
vData = dfr.getVolumeData()
vLap = None
vLapMed = None

fig, ax = pylab.subplots()
pylab.subplots_adjust(bottom=0.20)

sp1 = pylab.subplot(131)
sp2 = pylab.subplot(132)
sp3 = pylab.subplot(133)
    
#axes: left, bottom, width, height
slSlider = Slider(pylab.axes([0.1, 0.10, 0.8, 0.03]), 'Slice', 0, dfr.getNbSlices()-1, 0, valfmt='%1.0f')
siSlider = Slider(pylab.axes([0.1, 0.05, 0.8, 0.03]), 'Sigma', 1, 10, 2.5)

def update_slice(mySlice):
    mySlice = int(mySlice)
    sData = vData[:,:,mySlice]
    sLap = vLap[:,:,mySlice]
    sLapMed = vLapMed[:,:,mySlice]
    
    sp1.clear()
    sp2.clear()
    sp3.clear()
    
    sp1.set_title("Slice {}".format(mySlice))
    sp2.set_title("Laplacian")
    sp3.set_title("Ignore")
    sp1.imshow(sData, cmap=pylab.cm.bone)  # @UndefinedVariable
    sp2.imshow(sLap, cmap=pylab.cm.jet)  # @UndefinedVariable
    sp3.imshow(sLapMed, cmap=pylab.cm.jet)  # @UndefinedVariable
    fig.canvas.draw()
    
def update_sigma(sigma):
    sp2.set_title("Updating...")
    fig.canvas.draw()
    global vLap, vLapMed
    sigmas = np.array([sigma, sigma, sigma]) / voxelShape
    print "sigma: ", sigmas
    vLap = nd.filters.gaussian_laplace(vData, sigmas)
    print vLap.min(), vLap.max()
    #vLapMed = np.bitwise_and(vLap < 10, vLap > -10)
    vLapMed = generic_gradient_magnitude(mask3D, sobel).astype(float)
    vLapMed = nd.filters.gaussian_filter(vLapMed, 4.5)
    
    update_slice(slSlider.val)

update_sigma(5.0)

siSlider.on_changed(update_sigma)
slSlider.on_changed(update_slice)

pylab.show()