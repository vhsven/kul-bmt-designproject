import pylab
import numpy as np
import numpy.ma as ma
import scipy.ndimage as nd
import scipy.ndimage.morphology as morph
from matplotlib.widgets import Slider
from DicomFolderReader import DicomFolderReader
from XmlAnnotationReader import XmlAnnotationReader
from Preprocessor import Preprocessor

setID = int(raw_input("Load dataset #: "))
dfr = DicomFolderReader.create("../data/LIDC-IDRI", setID)
cc = dfr.getCoordinateConverter()
reader = XmlAnnotationReader(dfr.Path, cc)
for c, r in reader.getNodulePositions():
    print c, r
vh,vw,vd = dfr.getVoxelShape()
voxelShape = np.array([vh,vw,vd])
mask3D = Preprocessor.loadThresholdMask(setID)
vData = dfr.getVolumeData()
vLap = None
vEdges = morph.distance_transform_cdt(mask3D, metric='taxicab').astype(np.float32)

fig, ax = pylab.subplots()
pylab.subplots_adjust(bottom=0.20)

sp1 = pylab.subplot(131)
sp2 = pylab.subplot(132)
sp3 = pylab.subplot(133)
    
#axes: left, bottom, width, height
slSlider = Slider(pylab.axes([0.1, 0.10, 0.8, 0.03]), 'Slice', 0, dfr.getNbSlices()-1, 50, valfmt='%1.0f')
siSlider = Slider(pylab.axes([0.1, 0.05, 0.8, 0.03]), 'Sigma', 1, 30, 2.5)

def update_slice(mySlice):
    mySlice = int(mySlice)
    sMask = mask3D[:,:,mySlice]
    sData = vData[:,:,mySlice]
    #sData[~sMask] = sData.min()
    msData = ma.masked_array(sData, mask=~sMask)
    sLap = vLap[:,:,mySlice]
    msLap = ma.masked_array(sLap, mask=~sMask)
    sEdges = vEdges[:,:,mySlice]
    
    sp1.clear()
    sp2.clear()
    sp3.clear()
    
    sp1.set_title("Slice {}".format(mySlice))
    sp2.set_title("Laplacian")
    sp3.set_title("Edges")
    sp1.imshow(msData, cmap=pylab.cm.bone)  # @UndefinedVariable
    sp2.imshow(msLap, cmap=pylab.cm.jet)  # @UndefinedVariable
    sp3.imshow(sEdges, cmap=pylab.cm.jet)  # @UndefinedVariable
    fig.canvas.draw()
    
def update_sigma(sigma):
    sp2.set_title("Updating...")
    fig.canvas.draw()
    global vLap
    sigmas = np.array([sigma, sigma, sigma]) / voxelShape
    print "sigma: ", sigmas
    vLap = nd.filters.gaussian_laplace(vData, sigmas)
    
    #vLapMed = generic_gradient_magnitude(mask3D, sobel)
    #vLapMed = nd.filters.gaussian_filter(vLapMed, 4.5)
    
    update_slice(slSlider.val)

update_sigma(5.0)

siSlider.on_changed(update_sigma)
slSlider.on_changed(update_slice)

pylab.show()