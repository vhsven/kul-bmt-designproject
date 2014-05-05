'''
Created on 5-mei-2014

@author: Eigenaar
'''
from DicomFolderReader import DicomFolderReader
from XmlAnnotationReader import XmlAnnotationReader
'''
Created on 27-mrt.-2014

@author: Eigenaar
'''
import numpy as np
import scipy.ndimage as ndimage
from Constants import ProbTreshold


#######################################
# CLUSTERING OF DATA
########################################
def ClusteringData(probImage,scannumber):
        
    mask = probImage > ProbTreshold # zelf in te stellen threshold
    
    #clusterdata
    label_im,_ = ndimage.label(mask)
    
    # Compute size, mean_value, etc. of each region:
    #sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    #mean_vals = ndimage.sum(im, label_im, range(1, nb_labels + 1))
    
    # Find enclosing object:
    BB = ndimage.find_objects(label_im) # provides array of tuples: 3 tuples in 3D per bounding box (3 hoekpunten)
    NodGeg = []
    Nr = scannumber
    for i in range(0,len(BB)-1):
        bb = BB[i]
        point1 = [ f.start for f in bb]
        point2 = [ f.stop for f in bb]
        
        # centre of gravity
        xm = (point1[0] + point2[0]) / 2
        ym = (point1[1] + point2[1]) / 2
        zm = (point1[2] + point2[2]) / 2
        
        # mean probability
        AllProb = probImage[point1[0]:point2[0], point1[1]:point2[1],point1[2]:point2[2]]
        Mprob = AllProb.mean()
        
        
        if len(NodGeg) == 0:
            NodGeg = [Nr,xm,ym,zm,Mprob]
        else:
            NodGeg = np.vstack((NodGeg, [Nr,xm,ym,zm,Mprob]))
        
        # return all nodules + Mprob per nodule
    return(NodGeg)

################################################
    
##############################################
# VALIDATION
#############################################
def ValidateData(self, NodGeg):
    # INPUT
    # array with scannumber, 3D position (x,y,z), probability p
    #
    # 3D position= centre of gravity from bounding box
    # probability= mean of all probabilities of voxels in bouding box
    
    # SCRIPT (van ginneken)
    # is 'hit' situated in region around nodule centre (annotatie, radius 1,5)? yes, then TP
    # is 'hit' not situated in region? then FP
    
    NodGegTF=[]
    dfr = DicomFolderReader.create("../data/LIDC-IDRI", 1)
    cc = dfr.getCoordinateConverter()
    reader = XmlAnnotationReader("../data/LIDC-IDRI", cc)
    
    #for c,r in reader.getNodulePositions(): #this only works if we use world coordinates, and the nodules are nice spheres
    for i in range(0,len(NodGeg)-1):
        #NodGeg = Nr,xm,ym,zm,Mprob
        v=NodGeg[i,1:3]
        
        # amount of nodules
        InitAmountN = len(reader.Nodules)
        RestAm = InitAmountN
        
        # amount of FP
        FalseP = 0
        
        lijstje = []
    
        for c,r in reader.getNodulePositions():
            if sum((v-c)**2) < (2*r)**2:
                NodGegTF = np.hstack((NodGeg, 'TP'))
                
                if len(lijstje) == 0:
                    lijstje = [c]
                    RestAm -= 1
                elif c not in lijstje:
                    lijstje = np.vstack((lijstje,c))
                    RestAm -= 1
                                  
            else:
                NodGegTF = np.hstack((NodGeg, 'FP'))
                FalseP += 1
    
    AmountTP = InitAmountN - RestAm
    AmountFP = FalseP
    AmountFN = RestAm
    
    return NodGegTF, AmountTP, AmountFP, AmountFN
                

# sensitivity = (aantal echt positieven) / (aantal echt positieven + aantal fout negatieven)
# specificity = (aantal echt negatieven) / (aantal echt negatieven + aantal fout negatieven)

#######################################
        
        



    
