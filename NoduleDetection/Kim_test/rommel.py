import numpy as np
import scipy as sp
import scipy.ndimage as nd
import pylab
import numpy.ma as ma
from skimage.filter.rank import entropy
from skimage.morphology import disk
from DicomFolderReader import DicomFolderReader

 
myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
dfr = DicomFolderReader(myPath)
ds = dfr.Slices[87]
data = ds.pixel_array #voxel(i,j) is pixel(j,i) -> so one voxel is one pixel (http://nipy.org/nibabel/dicom/dicom_orientation.html)

def pixelentropy(data):
        # calculates the sliceEntropy of each pixel in the slice in comparison to its surroundings
        image = data[:,:]
        pylab.imshow(image, cmap=pylab.gray())
        pylab.show()
        print(image.shape)
        image0 = image
        pixelentr0=entropy(image0, disk(5))
        pylab.subplot(131)
        pylab.imshow(pixelentr0)
        
        image1 = image.view('uint8')
        print(type(image1))
        print(image1.shape)
        pixelentr1=entropy(image1, disk(5))
        pylab.subplot(132)
        pylab.imshow(pixelentr1)
    #     print(imentropy)
    #     img1 = ax1.imshow(imentropy, cmap=plt.cm.jet)
    #     ax1.set_title('Entropy')
    #     ax1.axis('off')
    #     plt.colorbar(img1, ax=ax1)
    #     plt.show()
    
        image2 = image.view('uint8')
        image2 = image2[:, 1::2]
        print(image2.shape)
        pixelentr2 = entropy(image2, disk(5))
        pylab.subplot(133)
        pylab.imshow(pixelentr2)
        pylab.show()
        
               
        return pixelentr0, pixelentr1, pixelentr2 #returns a matrix with sliceEntropy values for each pixel


from scipy.ndimage.filters import generic_gradient_magnitude, sobel

def getEdges(data):
        #import scipy
        #from scipy import ndimage
        #from scipy.ndimage.filters import generic_gradient_magnitude, sobel
                           
    #     dx = ndimage.sobel(self.Data, 0)  # x derivative
    #     dy = ndimage.sobel(self.Data, 1)  # y derivative
    #     dz = ndimage.sobel(self.Data, 2)  # z derivative
        
        mag = generic_gradient_magnitude(data, sobel)
            
        return mag
mag=getEdges(data)
pylab.imshow(mag)
pylab.show()

# data = np.ones((3,8,8))
# data = data*3
# data[1,2,1]=555
# data[1,2,0] = 60
# data[1,2,2] = 50
# data[0,2,1] = 30
# data[2,2,1] = 40
# data[1,1,1] = 1
# data[1,3,1] = 100