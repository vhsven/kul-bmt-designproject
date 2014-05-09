from XmlAnnotationReader import XmlAnnotationReader
from Constants import PROBABILITY_THRESHOLD
from scipy import ndimage
import numpy as np

class Validator:
    def __init__(self, myPath, cc):
        self.Path = myPath
        self.cc = cc
        
    def ClusteringData(self, probImg, setID):
        
        mask = probImg > PROBABILITY_THRESHOLD 
        
        #clusterdata
        label_im,_ = ndimage.label(mask)
        
        # Find enclosing object:
        BB = ndimage.find_objects(label_im) # provides array of tuples: 3 tuples in 3D per bounding box (3 hoekpunten)
        clusterData = []
        nbDiscarded = 0
        for bb in BB:
            probWindow = probImg[bb]
            if probWindow.shape <= (1,1,1):
                nbDiscarded += 1
                continue
            
            point1 = [ f.start for f in bb]
            point2 = [ f.stop for f in bb]
            
            # centre of gravity
            xm = (point1[0] + point2[0]) // 2
            ym = (point1[1] + point2[1]) // 2
            zm = (point1[2] + point2[2]) // 2
            
            # mean probability
            #probWindow = probImg[point1[0]:point2[0], point1[1]:point2[1],point1[2]:point2[2]]
            meanProb = probWindow.mean()
            
            
            if len(clusterData) == 0:
                clusterData = [setID,xm,ym,zm,meanProb]
            else:
                clusterData = np.vstack((clusterData, [setID,xm,ym,zm,meanProb]))
            
        print("Discarded {} 1px clusters.".format(nbDiscarded))   
        return clusterData

        
    def ValidateData(self, clusterData):
        """
        Input
        -----
        Array with setID, 3D position (x,y,z), probability p
          -> 3D position= centre of gravity from bounding box
          -> probability= mean of all probabilities of voxels in bouding box
        """
        reader = XmlAnnotationReader(self.Path, self.cc)
        foundNodules = set()
        nbNodules = len(reader.Nodules)
        nbFP = 0
        
        #for c,r in reader.getNodulePositions(): #this only works if we use world coordinates, and the nodules are nice spheres
        for nodule in reader.Nodules:
            regionCenters, regionRs = nodule.Regions.getRegionCenters()
            for pixelZ in regionCenters.keys():
                c = regionCenters[pixelZ]
                r = regionRs[pixelZ]
                for cluster in clusterData:
                    #clusterData = setID,xm,ym,zm,meanProb
                    v=cluster[1:3] #xm,ym
                    
                    if sum((v-c)**2) < (2*r)**2:
                        foundNodules.add(nodule)
                    else:
                        nbFP += 1
                                          
        nbTP = len(foundNodules) #nbNodulesFound
        nbFN = nbNodules - nbTP #nbNodulesLost
        
        return foundNodules, nbTP, nbFP, nbFN