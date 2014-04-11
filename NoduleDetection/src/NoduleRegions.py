import numpy as np
import pylab
from matplotlib.path import Path
import matplotlib.patches as patches
from Constants import MIN_NODULE_RADIUS

class NoduleRegions:
    def __init__(self):
        self.regions = {}
    
    def addRegion(self, pixelZ, coords):
        self.regions[pixelZ] = coords
        
    def getRegionCoords(self, pixelZ):
        return self.regions[pixelZ]
        
    def getNbRegions(self): 
        return len(self.regions.keys())
    
    def getSortedZIndices(self):
        return sorted(self.regions.iterkeys())
    
    def getRegionsSorted(self):
        allRegions = {}
        for pixelZ in self.getSortedZIndices():
            allRegions[pixelZ] = self.getRegionCoords(pixelZ)
        
        return allRegions
    
    def getRegionPath(self, pixelZ):
        coords = self.getRegionCoords(pixelZ)
        if len(coords) > 1:
            verts = [(x,y) for (x,y,_) in coords]
            codes = [Path.MOVETO] + [Path.LINETO] * (len(coords)-2) + [Path.CLOSEPOLY]
            path = Path(verts, codes)
            
            fig = pylab.figure()
            ax = fig.add_subplot(111)
            patch = patches.PathPatch(path, facecolor='black', lw=0)
            ax.add_patch(patch)
            ax.set_xlim(0,512)
            ax.set_ylim(0,512)
            pylab.show()
            
            x = np.arange(0, 512)
            y = np.arange(0, 512)
            path.contains_points()
            
            return path
        else:
            return None
    
    def getRegionMasks(self):
        masks = {}
        c = {}
        r2 = {}
        for pixelZ in self.getSortedZIndices():
            coords = self.getRegionCoords(pixelZ)
            x,y,_ = zip(*coords)
            x = np.array(x)
            y = np.array(y)
            #z = np.array(z)
            minX, maxX = min(x), max(x)
            minY, maxY = min(y), max(y)
            #minZ, maxZ = min(z), max(z)
            centerX = (maxX + minX) / 2
            centerY = (maxY + minY) / 2
            #centerZ = (maxZ + minZ) / 2
            c[pixelZ] = centerX, centerY
            r2[pixelZ] = max((x-centerX)**2 + (y-centerY)**2)
            if r2[pixelZ] < MIN_NODULE_RADIUS:
                r2[pixelZ] = MIN_NODULE_RADIUS
            
            # TODO get slice dimensions from somewhere
            masks[pixelZ] = np.zeros(512**2).reshape(512,512).astype(np.bool)
            x = np.arange(0, 512)
            y = np.arange(0, 512)
            dx = (x-centerX)**2
            dy = (y-centerY)**2
            
            # can we make this more efficient?
            for x in range(0, 512):
                for y in range(0, 512):
                    masks[pixelZ][y,x] = (dx[x] + dy[y] <= r2[pixelZ])            
        
        return masks, c, r2
    
    def printRegions(self):
        print("Found {0} regions.".format(self.getNbRegions()))
        for pixelZ in self.getSortedZIndices():
            coords = self.getRegionCoords(pixelZ)
            #print("\t\tFound {0} coordinates for region with pixelZ={1}:".format(len(coords), pixelZ))
            for coord in coords:
                print("{0} {1} {2}".format(coord[0], coord[1], coord[2]))