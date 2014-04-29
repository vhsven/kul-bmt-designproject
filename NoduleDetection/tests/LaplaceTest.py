import pylab
#import numpy as np
import scipy.ndimage as nd
from matplotlib.widgets import Slider
from DicomFolderReader import DicomFolderReader
from XmlAnnotationReader import XmlAnnotationReader

setID = int(raw_input("Load dataset #: "))
myPath = DicomFolderReader.findPath("../data/LIDC-IDRI", setID)
dfr = DicomFolderReader(myPath)
cc = dfr.getCoordinateConverter()
reader = XmlAnnotationReader(myPath, cc)
for c, r in reader.getNodulePositions():
    print c, r
vData = dfr.getVolumeData()
vLap = None

fig, ax = pylab.subplots()
pylab.subplots_adjust(bottom=0.20)

sp1 = pylab.subplot(121)
sp2 = pylab.subplot(122)
    
#axes: left, bottom, width, height
slSlider = Slider(pylab.axes([0.1, 0.10, 0.8, 0.03]), 'Slice', 0, dfr.getNbSlices()-1, 0, valfmt='%1.0f')
siSlider = Slider(pylab.axes([0.1, 0.05, 0.8, 0.03]), 'Sigma', 1, 10, 2.5)

def update_slice(mySlice):
    mySlice = int(mySlice)
    sData = vData[:,:,mySlice]
    sLap = vLap[:,:,mySlice]
    
    sp1.clear()
    sp2.clear()
    
    sp1.set_title("Slice {}".format(mySlice))
    sp2.set_title("Laplacian")
    sp1.imshow(sData, cmap=pylab.gray())
    sp2.imshow(sLap, cmap=pylab.gray())
    fig.canvas.draw_idle()
    
def update_sigma(sigma):
    sp2.set_title("Updating...")
    fig.canvas.draw()
    global vLap
    vLap = nd.filters.gaussian_laplace(vData, sigma)
    vLap = abs(vLap)
    
    update_slice(slSlider.val)

update_sigma(5.0)

siSlider.on_changed(update_sigma)
slSlider.on_changed(update_slice)

pylab.show()

