import pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport @UnusedImport works just fine
from DicomFolderReader import DicomFolderReader
from CoordinateConverter import CoordinateConverter

myPath = DicomFolderReader.findPathByID("../data/LIDC-IDRI/", 1)
dfr = DicomFolderReader(myPath)
matrix = dfr.getWorldMatrix()
cc = CoordinateConverter(matrix)
finder = dfr.getPixelFinder()
shape = dfr.getVolumeShape()
print(matrix)
dfr.printInfo()
 
pixels = list(finder.findNodulePixels(shape, method='polygon', radiusFactor=1.0))
pixelsWorld = list(cc.getWorldVectors(pixels))
X,Y,Z,_ = zip(*pixelsWorld)
 
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z, c='r', marker='o')
ax.set_title("Nodule Sphere Test")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#0001
ax.set_xlim([70,110])
ax.set_ylim([30,70])
ax.set_zlim([-90,-130])

#0007
#ax.set_xlim([10,50])
#ax.set_ylim([-10,-50])
#ax.set_zlim([-70,-110])
 
pl.show()
