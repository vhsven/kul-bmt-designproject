import numpy as np
import pylab
from matplotlib.path import Path
import matplotlib.patches as patches
from Constants import MIN_NODULE_RADIUS, NODULE_RADIUS_FACTOR

class NoduleRegions:
    def __init__(self):
        self.regions = {}
    
    def addRegion(self, z, coords):
        self.regions[z] = coords
        
    def getRegionCoords(self, z):
        return self.regions[z]
        
    def getNbRegions(self): 
        return len(self.regions.keys())
    
    def getSortedZIndices(self):
        return sorted(self.regions.iterkeys())
    
    def getRegionsSorted(self):
        allRegions = {}
        for z in self.getSortedZIndices():
            allRegions[z] = self.getRegionCoords(z)
        
        return allRegions
    
    
    def getRegionMasksPolygon(self):
        paths = {}
        masks = {}
        for z in self.getSortedZIndices():
            coords = self.getRegionCoords(z)
            if len(coords) > 1:
                verts = [(x,y) for (x,y,_) in coords]
                codes = [Path.MOVETO] + [Path.LINETO] * (len(coords)-2) + [Path.CLOSEPOLY]
                paths[z] = Path(verts, codes)
                
                x, y = np.meshgrid(np.arange(512), np.arange(512))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x,y)).T # array([[0, 0],[1, 0],[2, 0],[3, 0],...,[9, 0],[0, 1],[1, 1],...
                
                masks[z] = paths[z].contains_points(points)
                masks[z] = masks[z].reshape(512,512)
                
        return paths, masks

    
    def getRegionCenters(self):
        centers = {}
        r2 = {}
        for z in self.getSortedZIndices():
            coords = self.getRegionCoords(z)
            x,y,_ = zip(*coords)
            x = np.array(x)
            y = np.array(y)
            centerX = x.mean()
            centerY = y.mean()
            centers[z] = centerX, centerY
            rx = (x-centerX)**2
            ry = (y-centerY)**2
            r2[z] = max(rx + ry)
            
            r2[z] *= NODULE_RADIUS_FACTOR
            if r2[z] < MIN_NODULE_RADIUS:
                r2[z] = MIN_NODULE_RADIUS
            
        return centers, r2 #returns (r^2)/3 instead of r^2
        
    def getRegionMasksCircle(self):
        masks = {}
        c, r2 = self.getRegionCenters()
        for z in self.getSortedZIndices():
            centerX, centerY = c[z]
               
            # TODO get slice dimensions from somewhere
            masks[z] = np.zeros(512**2).reshape(512,512).astype(np.bool)
            x = np.arange(0, 512)
            y = np.arange(0, 512)
            dx = (x-centerX)**2
            dy = (y-centerY)**2
            
            # can we make this more efficient?
            for x in range(0, 512):
                for y in range(0, 512):
                    masks[z][y,x] = (dx[x] + dy[y] <= r2[z])            
        
        return masks, c, r2
    
#     def getRegionMasksSphere(self, cc):
#         masks = {}
#         c = {}
#         r2 = {}
#         coords = []
#         for z in self.getSortedZIndices():
#             coords += self.getRegionCoords(z)
#         
#         coords = [list(cc.getWorldVector(pixelVector)) for pixelVector in coords]
#         x,y,z,_ = zip(*coords) 
#         minX, maxX = min(x), max(x)
#         minY, maxY = min(y), max(y)
#         minZ, maxZ = min(z), max(z)
#         centerX = (maxX + minX) / 2
#         centerY = (maxY + minY) / 2
#         centerZ = (maxZ + minZ) / 2
#         rx = (x-centerX)**2
#         ry = (y-centerY)**2
#         rz = (z-centerZ)**2 
#         print max(rx), max(ry), max(rz) #max rz still much smaller than rx and ry
#         r2 = max(rx + ry + rz)
#         if r2 < MIN_NODULE_RADIUS:
#             r2 = MIN_NODULE_RADIUS
#         
#         # TODO get volume dimensions from somewhere
#         h,w,d = 512,512,100
#         #h,w,d = 50,50,50
#         mask = np.zeros(h*w*d).reshape(h,w,d).astype(np.bool)
#         
#         # we hebben tijd...
#         for px in range(0, h):
#             for py in range(0, w):
#                 for pz in range(0, d):
#                     worldVector = list(cc.getWorldVector([px, py, pz]))
#                     mask[px,py,pz] = ((worldVector[0]-centerX)**2 + (worldVector[1]-centerY)**2 + (worldVector[2]-centerZ)**2 <= r2)            
#         
#         return mask
    
    def printRegions(self):
        print("Found {0} regions.".format(self.getNbRegions()))
        for z in self.getSortedZIndices():
            coords = self.getRegionCoords(z)
            #print("\t\tFound {0} coordinates for region with z={1}:".format(len(coords), z))
            for coord in coords:
                print("{0} {1} {2}".format(coord[0], coord[1], coord[2]))
