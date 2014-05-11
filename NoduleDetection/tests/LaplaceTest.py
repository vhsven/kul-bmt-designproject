import pylab
import numpy as np
import numpy.ma as ma
import scipy.ndimage as nd
from matplotlib.widgets import Slider
from DicomFolderReader import DicomFolderReader
from Preprocessor import Preprocessor

setID = int(raw_input("Load dataset #: "))
dfr = DicomFolderReader.create("../data/LIDC-IDRI", setID)
dfr.printInfo()
vh,vw,vd = dfr.getVoxelShape()
voxelShape = np.array([vh,vw,vd])
mask3D = Preprocessor.loadThresholdMask(setID)
vData = dfr.getVolumeData()
vLap = None

# fig, ax = pylab.subplots()
# pylab.subplots_adjust(bottom=0.20)
# 
# sp1 = pylab.subplot(121)
# sp2 = pylab.subplot(122)
#     
# #axes: left, bottom, width, height
# slSlider = Slider(pylab.axes([0.1, 0.10, 0.8, 0.03]), 'Slice', 0, dfr.getNbSlices()-1, 65, valfmt='%1.0f')
# #siSlider = Slider(pylab.axes([0.1, 0.05, 0.8, 0.03]), 'Sigma', 1, 30, 2.5)
# 
# def update_slice(mySlice):
#     mySlice = int(mySlice)
#     sMask = mask3D[:,:,mySlice]
#     sData = vData[:,:,mySlice]
#     #sData[~sMask] = sData.min()
#     msData = ma.masked_array(sData, mask=~sMask)
#     sLap = vLap[:,:,mySlice]
#     msLap = ma.masked_array(sLap, mask=~sMask)
#     
#     sp1.clear()
#     sp2.clear()
#     
#     sp1.set_title("Slice {}".format(mySlice))
#     sp2.set_title("Laplacian")
#     sp1.imshow(msData, cmap=pylab.cm.bone)  # @UndefinedVariable
#     sp2.imshow(msLap, cmap=pylab.cm.jet)  # @UndefinedVariable
#     fig.canvas.draw()
#     
# def update_sigma(sigma):
#     sp2.set_title("Updating...")
#     fig.canvas.draw()
#     global vLap
#     sigmas = np.array([sigma, sigma, sigma]) / voxelShape #convert to pixels
#     print "sigma: ", sigmas
#     vLap = nd.filters.gaussian_laplace(vData, sigmas)
#     vLap[vLap > 0] = 0
#     print vLap.min()
#     
#     update_slice(slSlider.val)
# 
# 
# #siSlider.on_changed(update_sigma)
# slSlider.on_changed(update_slice)

mySlice = 89
sMask = mask3D[:,:,mySlice]
for sigma in np.arange(1, 10):
    sigmas = np.array([sigma, sigma, sigma]) / voxelShape #convert to pixels
    vLap = nd.filters.gaussian_laplace(vData, sigmas) #TODO normalize: *t
    vLap[vLap > 0] = 0
    vLap *= -1
    
    sLap = vLap[:,:,mySlice]
    msLap = ma.masked_array(sLap, mask=~sMask)
    
    fig = pylab.figure()
    ax = pylab.imshow(msLap, pylab.cm.bone)  # @UndefinedVariable
    fig.colorbar(ax)
    pylab.title("Laplacian ($\sigma$ = {})".format(sigmas))
    pylab.show()