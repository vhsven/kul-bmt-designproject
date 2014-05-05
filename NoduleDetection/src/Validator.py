from XmlAnnotationReader import XmlAnnotationReader
from Constants import ProbTreshold
from scipy import ndimage
import numpy as np

class Validator:
    def __init__(self, myPath, cc):
        self.Path = myPath
        self.cc = cc
        
    def ClusteringData(self, probImage, scannumber):
        
        mask = probImage > ProbTreshold 
        
        #clusterdata
        label_im,_ = ndimage.label(mask)
        
        # Find enclosing object:
        BB = ndimage.find_objects(label_im) # provides array of tuples: 3 tuples in 3D per bounding box (3 hoekpunten)
        NodGeg = []
        Nr = scannumber
        for bb in BB:
            point1 = [ f.start for f in bb]
            point2 = [ f.stop for f in bb]
            
            # centre of gravity
            xm = (point1[0] + point2[0]) // 2
            ym = (point1[1] + point2[1]) // 2
            zm = (point1[2] + point2[2]) // 2
            
            # mean probability
            AllProb = probImage[point1[0]:point2[0], point1[1]:point2[1],point1[2]:point2[2]]
            Mprob = AllProb.mean()
            
            
            if len(NodGeg) == 0:
                NodGeg = [Nr,xm,ym,zm,Mprob]
            else:
                NodGeg = np.vstack((NodGeg, [Nr,xm,ym,zm,Mprob]))
            
            
        return NodGeg

        
    def ValidateData(self, NodGeg):
        # INPUT
        # array with scannumber, 3D position (x,y,z), probability p
        #
        # 3D position= centre of gravity from bounding box
        # probability= mean of all probabilities of voxels in bouding box
        
        #dfr = DicomFolderReader.create("../data/LIDC-IDRI", 1)
        #cc = dfr.getCoordinateConverter()
        reader = XmlAnnotationReader(self.Path, self.cc)
        
        # amount of nodules
        InitAmountN = len(reader.Nodules)
        RestAm = InitAmountN
        
        # amount of FP
        FalseP = 0
        
        lijstje = list()
        NodGegT = []
        NodGegF = []
        
        #for c,r in reader.getNodulePositions(): #this only works if we use world coordinates, and the nodules are nice spheres
        for nodule in reader.Nodules:
            print(nodule.ID)
            regionCenters, regionRs = nodule.Regions.getRegionCenters()
            for pixelZ in regionCenters.keys():
                c = regionCenters[pixelZ]
                r = regionRs[pixelZ]
                print(len(NodGeg))
                for cluster in NodGeg:
                    #NodGeg = Nr,xm,ym,zm,Mprob
                    v=cluster[1:3]
                    
                                            
                    if sum((v-c)**2) < (4*r)**2:
                                                                        
                        if nodule not in lijstje:
                            lijstje.append(nodule)
                            RestAm -= 1
                            
                        if len(NodGegT) == 0:
                            NodGegT = cluster
                        else:
                            NodGegT = np.vstack((NodGegT, cluster))
                                          
                    else:
                        FalseP += 1
                        if len(NodGegF) == 0:
                            NodGegF = cluster
                        else:
                            NodGegF = np.vstack((NodGegF, cluster))
                                          
        
        AmountTP = InitAmountN - RestAm
        AmountFP = FalseP
        AmountFN = RestAm
        
        return NodGegT, NodGegF, lijstje, AmountTP, AmountFP, AmountFN
