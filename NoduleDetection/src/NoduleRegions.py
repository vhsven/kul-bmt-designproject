import math
import numpy as np
from matplotlib.path import Path
#from Constants import MIN_NODULE_RADIUS

class NoduleRegions:
    def __init__(self):
        self.Regions = {}
    
    def __del__(self):
        del self.Regions
        
    def addRegion(self, z, coords):
        self.Regions[z] = coords
        
    def getRegionCoords(self, z):
        return self.Regions[z]
        
    def getNbRegions(self): 
        return len(self.Regions.keys())
    
    def getSortedZIndices(self):
        return sorted(self.Regions.iterkeys())
    
    def getRegionsSorted(self):
        allRegions = {}
        for z in self.getSortedZIndices():
            allRegions[z] = self.getRegionCoords(z)
        
        return allRegions
    
    def getRegionMasksPolygon(self, m, n):
        paths = {}
        masks = {}
        for z in self.getSortedZIndices():
            coords = self.getRegionCoords(z)
            if len(coords) > 1:
                verts = [(x,y) for (x,y,_) in coords]
                codes = [Path.MOVETO] + [Path.LINETO] * (len(coords)-2) + [Path.CLOSEPOLY]
                paths[z] = Path(verts, codes)
                
                x, y = np.meshgrid(np.arange(m), np.arange(n))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x,y)).T # array([[0, 0],[1, 0],[2, 0],[3, 0],...,[9, 0],[0, 1],[1, 1],...
                
                masks[z] = paths[z].contains_points(points)
                masks[z] = masks[z].reshape(m,n)
                
        return paths, masks

    
    def getRegionCenters(self, minOrMax):
        centers = {}
        r = {}
        for z in self.getSortedZIndices():
            coords = self.getRegionCoords(z) #list of x,y,z tuples
            coords = np.array(coords)[:,[0,1]] #save x,y in numpy array
            centers[z] = coords.mean(axis=0)
            ssd = ((coords - centers[z]) ** 2).sum(axis=1)
            r[z] = math.sqrt(minOrMax(ssd))
        
        return centers, r
        
    def getRegionMasksCircle(self, m, n, minOrMax, radiusFactor=1.0):
        masks = {}
        centers, r = self.getRegionCenters(minOrMax)
        for z in self.getSortedZIndices():            
            x, y = np.meshgrid(np.arange(m), np.arange(n))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T # array([[0, 0],[1, 0],[2, 0],[3, 0],...,[9, 0],[0, 1],[1, 1],...
            masks[z] = ((points-centers[z])**2).sum(axis=1) <= (r[z] * radiusFactor)**2
            masks[z] = masks[z].reshape(m, n)
                
        return masks, centers, r
    
    def getRegionMask3D(self, shape, minOrMax, radiusFactor=1.0):
        h,w,_ = shape
        mask3D = np.zeros(shape, dtype=np.bool)
        masks, _, _ = self.getRegionMasksCircle(h, w, minOrMax, radiusFactor)
        for z in masks:
            mask3D[:,:,int(z)] = masks[z]
            
        return mask3D
    
    def isPointInsideCircles(self, p, minOrMax, radiusFactor=1.0):
        x,y,z = p
        z += 0.5 
        if not z in self.Regions:
            return False #no factor for top/bottom
        
        v = np.array([x,y])
        coords = [(c[0], c[1]) for c in self.getRegionCoords(z)] #list of x,y,z tuples -> x,y
        coords = np.array(coords)
        center = coords.mean(axis=0)
        ssd = ((coords - center) ** 2).sum(axis=1)
        r = math.sqrt(minOrMax(ssd)) * radiusFactor
        return sum((v - center)**2) < r**2
    
#     def getRegionMasksSphere(self, h, w, d, cc):
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
