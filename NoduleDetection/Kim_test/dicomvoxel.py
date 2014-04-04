'''
Created on 3-apr.-2014

@author: Eigenaar
'''
import pylab
from DicomFolderReader import DicomFolderReader 

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/000000"
dfr = DicomFolderReader(myPath)
ds = dfr.Slices[50]
data = ds.pixel_array #voxel(i,j) is pixel(j,i) -> so one voxel is one pixel (http://nipy.org/nibabel/dicom/dicom_orientation.html)
#show image
#pylab.imshow(ds.pixel_array, cmap=pylab.gray())
#pylab.show()
print (data)

# for 1 pixel
featurevector=[]
#featurevector[1]=greyvalue
    # 2D Keschani et al
    
#featurevector[2]=intensity of that grey value

#featurevector[3]= 3D averaging (Keshani et al)

#featurevector[4]= position of voxel (regarding the lung wall?)

#featurevector[5]= window: trek links/R af van rechts/L -> verschillende groottes window

#featurevector[6]= window: divide left/R by right/L

#featurevector[7]= window: trek boven/onder af van onder/boven -> verschillende groottes window

#featurevector[8]= window: divide above/below by below/above

#featurevector[9]= features Ozekes and Osman (2010)

#featurevector[10]= 

