import pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport @UnusedImport works just fine
from FeatureGenerator import FeatureGenerator
from XmlAnnotationReader import XmlAnnotationReader
from PixelFinder import PixelFinder
from DicomFolderReader import DicomFolderReader

myPath = DicomFolderReader.findPath("../data/LIDC-IDRI/", 5) 
reader = XmlAnnotationReader(myPath)
matrix = reader.dfr.getWorldMatrix()
print(matrix)
cc = reader.dfr.getCoordinateConverter()
data = reader.dfr.getVolumeData()
#h,w,d = reader.dfr.getVolumeShape()
fgen = FeatureGenerator(data, reader.dfr.getVoxelShape())
finder = PixelFinder(reader)
 
pixels = list(finder.findNodulePixels(method='polygon', radiusFactor=1.0))
#x,y,z = zip(*pixels)
 
pixelsWorld = list(cc.getWorldVectors(pixels))
X,Y,Z,_ = zip(*pixelsWorld)
 
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z, c='r', marker='o')
ax.set_title("Nodule Sphere Test")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#ax.set_xlim([0,h])
#ax.set_ylim([0,w])
#ax.set_zlim([0,d])
ax.set_xlim([-235,100])
ax.set_ylim([-170,100])
ax.set_zlim([-350,350])
 
pl.show()
