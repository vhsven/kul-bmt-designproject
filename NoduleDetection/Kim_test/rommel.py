import numpy as np
import scipy as sp
import scipy.ndimage as nd
import pylab
import numpy.ma as ma
from DicomFolderReader import DicomFolderReader

def neighbours(x,y,z,data):
    # top - bottom neighbours
    Ptop = data[x,y-1,z]
    Pbottom = data[x,y+1,z]
    print(Ptop)
    print(Pbottom)
    
    Ptbmin = Ptop - Pbottom
    Ptbdiv = Ptop/Pbottom
    Ptbplus = Ptop + Pbottom
    
    Ppixeltopmin = data[x,y,z] - Ptop
    Ppixelbottommin = data[x,y,z] - Pbottom
    
    Ppixeltopplus = data[x,y,z] + Ptop
    Ppixelbottomplus = data[x,y,z] + Pbottom
    
    Ppixeltopdiv = data[x,y,z] / Ptop
    Ppixelbottomdiv = data[x,y,z] / Pbottom
    
        
    # left - right neighbours
    PL = data[x-1,y,z]
    PR = data[x+1,y,z]
    print(PL)
    print(PR)
    
    PLRmin = PL - PR
    PLRdiv = PL/PR
    PLRplus = PL + PR
    
    PpixelLmin = data[x,y,z] - PL
    PpixelRmin = data[x,y,z] - PR
    
    PpixelLplus = data[x,y,z] + PL
    PpixelRplus = data[x,y,z] + PR
    
    PpixelLdiv = data[x,y,z] / PL
    PpixelRdiv = data[x,y,z] / PR
    
        
    # front - back neighbours
    Pf = data[x,y,z-1]
    Pb = data[x,y,z+1]
    
    print(Pf)
    print(Pb)
    Pfbmin = Pf - Pb
    Pfbdiv = Pf/Pb
    Pfbplus = Pf + Pb
    
    Ppixelfmin = data[x,y,z] - Pf
    Ppixelbmin = data[x,y,z] - Pb
    
    Ppixelfplus = data[x,y,z] + Pf
    Ppixelbplus = data[x,y,z] + Pb
    
    Ppixelfdiv = data[x,y,z] / Pf
    Ppixelbdiv = data[x,y,z] / Pb
    
    return Ptop, Pbottom, Ptbmin, Ptbdiv, Ptbplus, Ppixeltopmin, Ppixelbottommin, Ppixeltopplus, Ppixelbottomplus, Ppixeltopdiv, Ppixelbottomdiv, PL, PR, PLRmin, PLRdiv, PLRplus, PpixelLmin, PpixelRmin, PpixelLplus, PpixelRplus, PpixelLdiv, PpixelRdiv, Pf, Pb, Pfbmin, Pfbdiv, Pfbplus, Ppixelfmin, Ppixelbmin, Ppixelfplus, Ppixelbplus, Ppixelfdiv, Ppixelbdiv
def edges(x,y,data):
    import scipy
    from scipy import ndimage
        
    dx = ndimage.sobel(data, 0)  # x derivative
    dy = ndimage.sobel(data, 1)  # y derivative
    dz = ndimage.sobel(data, 2)  # z derivative
    

    return dx,dy,dz

 
myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
dfr = DicomFolderReader(myPath)
ds = dfr.Slices[87]
data = ds.pixel_array #voxel(i,j) is pixel(j,i) -> so one voxel is one pixel (http://nipy.org/nibabel/dicom/dicom_orientation.html)

LoG = nd.gaussian_laplace(data, 1.8) # scalar: standard deviations of the Gaussian filter
                                    # empirisch vastgesteld op 1.9/ 2/ 2.1
aLoG = abs(LoG)
output = np.copy(data)
output[aLoG > aLoG.max()-200] = 1

pylab.imshow(output, cmap=pylab.gray())
pylab.show()
  
data = np.ones((3,8,8))
data = data*3
data[1,2,1]=555
data[1,2,0] = 60
data[1,2,2] = 50
data[0,2,1] = 30
data[2,2,1] = 40
data[1,1,1] = 1
data[1,3,1] = 100