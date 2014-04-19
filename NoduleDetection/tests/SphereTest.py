import pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport @UnusedImport works just fine
from FeatureGenerator import FeatureGenerator
from XmlAnnotationReader import XmlAnnotationReader
from PixelFinder import PixelFinder

myPath = "../data/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/000000"
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
ax.set_xlim([70,110])
ax.set_ylim([30,70])
ax.set_zlim([-135,-95])

pl.show()
