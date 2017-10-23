from scipy import ndimage

class Validator:
    def __init__(self, nodules):
        self.Nodules = nodules
    
    def searchNodules(self, positives):
        nbTP = 0
        nbFN = 0
        for nodule in self.Nodules:
            mask3D = nodule.Regions.getRegionMask3D(positives.shape, max, radiusFactor=1.5)
            nbPositiveVoxels = positives[mask3D].sum()
            positives[mask3D] = 0 #clear this area
            #print nbPositiveVoxels
            if nbPositiveVoxels > 0:
                nbTP += 1
            else:
                nbFN += 1
                
        return nbTP, nbFN, positives
        
    def searchFPs(self, positives):        
        label_im,_ = ndimage.label(positives)
        BB = ndimage.find_objects(label_im) # provides array of tuples: 3 tuples in 3D per bounding box (3 hoekpunten)
#         clusteredData = []
        nbDiscarded = 0
        for bb in BB:
            probWindow = positives[bb]
            if probWindow.shape <= (1,1,1):
                nbDiscarded += 1
                continue
            
#             point1 = [ f.start for f in bb]
#             point2 = [ f.stop for f in bb]
#             
#             # centre of gravity
#             xm = (point1[0] + point2[0]) // 2
#             ym = (point1[1] + point2[1]) // 2
#             zm = (point1[2] + point2[2]) // 2
#             
#             # mean probability
#             #probWindow = probImg[point1[0]:point2[0], point1[1]:point2[1],point1[2]:point2[2]]
#             meanProb = probWindow.mean()
#             
#             
#             if len(clusteredData) == 0:
#                 clusteredData = [xm,ym,zm,meanProb]
#             else:
#                 clusteredData = np.vstack((clusteredData, [xm,ym,zm,meanProb]))
            
        print("Discarded {} 1px clusters.".format(nbDiscarded))
        nbFP = len(BB) - nbDiscarded   
        return nbFP
