import sys
import scipy.stats
import numpy as np
import math
import collections
from collections import deque
from skimage.filter.rank import entropy
from skimage.morphology import disk
import scipy.ndimage as nd
from scipy.ndimage.filters import generic_gradient_magnitude, sobel
from Preprocessor import Preprocessor

class FeatureGenerator:
    def __init__(self, setID, data, vshape, level=1):
        self.SetID = setID
        self.Data = data
        self.VoxelShape = vshape
        self.Level = level
        self.PixelCount = None
        
    def __str__(self):
        return "Level {} Feature Generator".format(self.Level)
    
    def __del__(self):
        del self.SetID
        del self.Data
        del self.VoxelShape
        del self.Level
        del self.PixelCount
        
    def getSlice(self, z):
        return self.Data[:,:,int(z)]
    
    # N: number of datapoints
    # l(i): total features per level i
    # L: total features over all levels
    def getLevelFeatureByMask(self, level, mask3D):
        """Returns a ndarray (Nxl) containing the features for all given positions and for the given level.""" 
        if level == 1:
            return self.getIntensityByMask(mask3D)
        if level == 2:
            N = mask3D.sum()
            start,stop = 2,6
            result = np.empty((N,stop-start+1))
            for sigma in np.arange(start,stop):
                sigmas = np.array([sigma]*3) / np.array(self.VoxelShape)
                print(sigmas)
                result[:,sigma-start] = self.getLaplacianByMask(mask3D, sigmas)
            result[:,stop-start] = self.getBlurredEdgesByMask(mask3D, self.SetID, sigma=4.5)
            return result
        else:
            print("Falling back on features per pixel method.")
            result = deque()
            for x,y,z in zip(np.where(mask3D)):
                feature = self.getLevelFeature(level, x, y, z)
                result.append(feature)
            return np.array(result).reshape((-1, 1))
        
    def getAllFeaturesByMask(self, mask3D):
        """Returns a ndarray (NxL) with the rows containing the features vector, up to the current level, per datapoint."""
        nbVoxels = mask3D.sum()
        h,w,d = mask3D.shape
        totalVoxels = h*w*d
        ratio = 100.0 * nbVoxels / totalVoxels
        print("Generating features for {0} ({1:.2f}%) voxels.".format(nbVoxels, ratio))
        allFeatures = self.getLevelFeatureByMask(1, mask3D)
        for level in range(2, self.Level+1):
            lvlFeature = self.getLevelFeatureByMask(level, mask3D) #Nxl
            allFeatures = np.hstack([allFeatures, lvlFeature])
        
        return allFeatures
    
#     def getLevelFeature(self, level, x,y,z):
#         """Returns a scalar representing the feature at the given position for the given level."""
#         if level == 1:
#             return self.getIntensity(x, y, z)
#         if level == 2: #TODO this is very inefficient
#             start,stop = 2,6
#             result = np.empty((1,stop-start+1))
#             for sigma in np.arange(start,stop):
#                 sigmas = np.array([sigma]*3) / np.array(self.VoxelShape)
#                 print sigmas
#                 result[:,sigma-start] = self.getLaplacian(x,y,z, sigmas)
#             result[:,stop-start] = self.getBlurredEdges(x,y,z, self.SetID, sigma=4.5)
#             return result
#         #if level == 3:
#         #    return self.getEntropy(x,y,z, windowSize=5)
#         #if level == 4:
#         #    return self.getEdges(x, y, z)
#         else:
#             raise ValueError("Level {} not supported.".format(level))
#     
#     def getAllFeatures(self, x,y,z): #TODO distance from lung wall?
#         """Returns a ndarray (1xL) containing the feature vector, up to the current level, for the given datapoint."""
#         z = int(z)
#         
#         allFeatures = self.getLevelFeature(1, x, y, z)
#         #allFeatures = deque()
#         for level in range(2, self.Level+1):
#             lvlFeature = self.getLevelFeature(level, x, y, z)
#             allFeatures = np.hstack([allFeatures, lvlFeature])
#             #allFeatures.append(lvlFeature)
#         
#         return np.array(allFeatures).reshape((1,-1))
# 
#     def getIntensity(self, x, y, z):
#         """Returns a scalar representing the intensity at the given position."""
#         return self.Data[x,y,z]
    
    def getIntensityByMask(self, mask3D):
        """Returns an array (Nx1) containing the intensities of all the given positions."""
        intensities = self.Data[mask3D]
        return intensities.reshape((-1, 1))
    
#     def getRelativePosition(self, x, y, z):
#         h,w,d = self.Data.shape
#         return float(x)/h, float(y)/w, float(z)/d
#     
#     def getRelativePositionByMask(self, mask3D):
#         h,w,d = self.Data.shape
#         xs, ys, zs = np.where(mask3D)
#         xsr = xs / float(h)
#         ysr = ys / float(w)
#         zsr = zs / float(d)
#         return np.vstack([xsr,ysr,zsr]).T
    
    def getLaplacianByMask(self, mask3D, sigmas):
        return nd.filters.gaussian_laplace(self.Data, sigmas)[mask3D]
            
    def getBlurredEdgesByMask(self, mask3D, setID, sigma=4.5):
        result = Preprocessor.loadThresholdMask(setID)
        result = generic_gradient_magnitude(result, sobel).astype(np.float32)
        result = nd.filters.gaussian_filter(result, sigma)
            
        return result[mask3D]
            
    ############################################################
    #featurevector[2]= greyvalue + related features in window
    ############################################################
    def greyvaluecharateristic(self, x,y,z,windowrowvalue):
        # windowrowvalue should be odd number (3,5,7...)
        
        # grey value
        greyvalue=self.Data[x,y,z]
        
        # square windowrowvalue x windowrowvalue
        valdown = windowrowvalue // 2
        valup   = valdown + 1
        
        windowD=self.Data[x-valdown:x+valup,y-valdown:y+valup,z-valdown:z+valup]
        
        #reshape window into array
        h,w,d=windowD.shape
        arrayD = np.reshape(windowD, (h*w*d))
        
        # mean and variance
        M=arrayD.mean()
        V=arrayD.var()
        
#         rangex = range(w)
#         rangey = range(h)
#         rangez = range(d)
    
    
        #calculate projections along the axes
#         xp = np.sum(windowD,axis=0)
#         yp = np.sum(windowD,axis=1)
#         zp = np.sum(windowD,axis=2)
    
        #centroid
#         cx = np.sum(rangex*xp)/np.sum(xp)
#         cy = np.sum(rangey*yp)/np.sum(yp)
#         cz = np.sum(rangez*zp)/np.sum(zp)
    
        #standard deviation
#         x2 = (rangex-cx)**2
#         y2 = (rangey-cy)**2
#         z2 = (rangez-cz)**2
#     
#         sx = np.sqrt( np.sum(x2*xp)/np.sum(xp) )
#         sy = np.sqrt( np.sum(y2*yp)/np.sum(yp) )
#         sz = np.sqrt( np.sum(z2*zp)/np.sum(zp) )
    
        #skewness
        skx = scipy.stats.skew(windowD, axis=0).mean()
        sky = scipy.stats.skew(windowD, axis=1).mean()
        skz = scipy.stats.skew(windowD, axis=2).mean()
        
        #x3 = (rangex-cx)**3
        #y3 = (rangey-cy)**3
        #z3 = (rangez-cz)**3
    
        #skx = np.sum(xp*x3)/(np.sum(xp) * sx**3)
        #sky = np.sum(yp*y3)/(np.sum(yp) * sy**3)
        #skz = np.sum(zp*z3)/(np.sum(zp) * sz**3)
    
        #Kurtosis
        kx = scipy.stats.kurtosis(windowD, axis=0).mean()
        ky = scipy.stats.kurtosis(windowD, axis=1).mean()
        kz = scipy.stats.kurtosis(windowD, axis=2).mean()
        
        #x4 = (rangex-cx)**4
        #y4 = (rangey-cy)**4
        #z4 = (rangez-cz)**4
        #kx = np.sum(xp*x4)/(np.sum(xp) * sx**4)
        #ky = np.sum(yp*y4)/(np.sum(yp) * sy**4)
        #kz = np.sum(zp*z4)/(np.sum(zp) * sz**4)
        
        #autocorrelation
        #result = np.correlate(arrayD, arrayD, mode='full')
        #autocorr=result[result.size/2:]
        
        # maximum and minimum greyvalue of pixels in window
        Max_greyvalue = arrayD.max()
        Min_greyvalue = arrayD.min()
        
        # difference between greyvalue pixel and max/min grey value
        maxdiff = abs(self.Data[x,y,z] - Max_greyvalue)
        mindiff = self.Data[x,y,z] - Min_greyvalue
        
        maxplus = self.Data[x,y,z] + Max_greyvalue
        minplus = self.Data[x,y,z] + Min_greyvalue
        
        maxdiv = self.Data[x,y,z]/Max_greyvalue
        mindiv = self.Data[x,y,z]/Min_greyvalue
        
        maxmindiff = Max_greyvalue - Min_greyvalue
        
        # count value pixel/max/min in window
        counter = collections.Counter(arrayD)
        freq_pixelvalue = counter[self.Data[x,y,z]] # prevalence of pixelvalue in window
        
        freq_max = counter[Max_greyvalue]
        freq_min = counter[Min_greyvalue]
        
        return  greyvalue,M,V,skx,sky,skz,kx,ky,kz,\
                Max_greyvalue,Min_greyvalue,maxdiff,mindiff,maxdiv,minplus,maxplus, mindiv,maxmindiff,\
                freq_pixelvalue,freq_max,freq_min #cx,cy,cz,sx,sy,sz
    
    def neighbours(self, x,y,z): #TODO zoals windowFeatures?
        # top - bottom neighbours
        Ptop = self.Data[x,y-1,z].astype('int32')
        Pbottom = self.Data[x,y+1,z].astype('int32')
        #print(type(Ptop), type(Ptop * Pbottom))
        #print(Ptop, Pbottom, Ptop*Pbottom)
        
        Ptbmin = Ptop - Pbottom
        Ptbdiv = Ptop*Pbottom
        Ptbplus = Ptop + Pbottom
        
        Ppixeltopmin = self.Data[x,y,z] - Ptop
        Ppixelbottommin = self.Data[x,y,z] - Pbottom
        
        Ppixeltopplus = self.Data[x,y,z] + Ptop
        Ppixelbottomplus = self.Data[x,y,z] + Pbottom
        
        Ppixeltopdiv = self.Data[x,y,z] * Ptop
        Ppixelbottomdiv = self.Data[x,y,z] * Pbottom
        
            
        # left - right neighbours
        PL = self.Data[x-1,y,z].astype('int32')
        PR = self.Data[x+1,y,z].astype('int32')
        
        PLRmin = PL - PR
        PLRdiv = PL*PR
        PLRplus = PL + PR
        
        PpixelLmin = self.Data[x,y,z] - PL
        PpixelRmin = self.Data[x,y,z] - PR
        
        PpixelLplus = self.Data[x,y,z] + PL
        PpixelRplus = self.Data[x,y,z] + PR
        
        PpixelLdiv = self.Data[x,y,z] * PL
        PpixelRdiv = self.Data[x,y,z] * PR
        
            
        # front - back neighbours
        Pf = self.Data[x,y,z-1].astype('int32')
        Pb = self.Data[x,y,z+1].astype('int32')
        Pfbmin = Pf - Pb
        Pfbdiv = Pf*Pb
        Pfbplus = Pf + Pb
        
        Ppixelfmin = self.Data[x,y,z] - Pf
        Ppixelbmin = self.Data[x,y,z] - Pb
        
        Ppixelfplus = self.Data[x,y,z] + Pf
        Ppixelbplus = self.Data[x,y,z] + Pb
        
        Ppixelfdiv = self.Data[x,y,z] * Pf
        Ppixelbdiv = self.Data[x,y,z] * Pb
        
        return  Ptop, Pbottom, Ptbmin, Ptbdiv, Ptbplus, Ppixeltopmin, Ppixelbottommin, Ppixeltopplus, Ppixelbottomplus, Ppixeltopdiv, Ppixelbottomdiv, \
                PL, PR, PLRmin, PLRdiv, PLRplus, PpixelLmin, PpixelRmin, PpixelLplus, PpixelRplus, PpixelLdiv, PpixelRdiv, \
                Pf, Pb, Pfbmin, Pfbdiv, Pfbplus, Ppixelfmin, Ppixelbmin, Ppixelfplus, Ppixelbplus, Ppixelfdiv, Ppixelbdiv
        
        
       
    ############################################################
    #featurevector[3]= prevalence of that grey value
    ############################################################
    def greyvaluefrequency(self, x,y,z):
        if self.PixelCount == None:
            self.PixelCount = collections.Counter(self.Data.ravel())
        
        pixelValue = self.Data[x,y,z]
        freqvalue = self.PixelCount[pixelValue] # prevalence of pixelvalue in image
        
        # prevalence maximum and minimum of pixels in image
        Max_image = self.Data.max()
        Min_image = self.Data.min()
        
        # prevalence max and min
        freqmax = self.PixelCount[Max_image]
        freqmin = self.PixelCount[Min_image]
        
        # compare (prevalence of) pixelvalue to min and max (prevalence)
        comfreq_max = freqvalue/freqmax
        comfreq_min = freqvalue/freqmin
        
        rel_max = self.Data[x,y,z]/Max_image
        rel_min = self.Data[x,y,z]/Min_image
        
        
        return freqvalue, comfreq_max, comfreq_min, rel_max, rel_min
    
    
    ############################################################
    #featurevector[4]=  frobenius norm pixel to center 2D image
    ############################################################
    def forbeniusnorm (self, x,y,z):
        # slice is 512 by 512 by numberz: b is center
        xb = 256
        yb = 256
        zb = self.Data.shape[2] // 2
        a = np.array((x,y,z))
        b = np.array((xb,yb,zb))
        dist = np.linalg.norm(a-b)
        
        return dist
    
    
    ############################################################
    #featurevector[5]= window: substraction L/R U/D F/B
    ############################################################
    def windowFeatures(self, x,y,z,windowrowvalue):
        valdown = windowrowvalue // 2
        valup   = valdown + 1
        
        windowD=self.Data[x-valdown:x+valup,y-valdown:y+valup,z-valdown:z+valup]
        
        # calculate 'getEdges' by substraction 
        leftrow=windowD[:,0,:]
        rightrow=windowD[:,windowrowvalue-1,:]
        meanL=leftrow.mean()
        meanR=rightrow.mean()
        gradLRmean=(rightrow-leftrow).mean()
        gradmeanLR=meanR-meanL
        
        # calculate 'getEdges' by division
        divmeanLR=meanR*meanL
        divLRmean=(leftrow*rightrow).mean()
        
        # calculate 'getEdges' by substraction 
        toprow=windowD[0,:,:]
        bottomrow=windowD[windowrowvalue-1, :, :]
        Tmean=toprow.mean()
        Bmean=bottomrow.mean()
        gradmeanUD=Tmean-Bmean
        gradUDmean=(toprow-bottomrow).mean()
        
        # calculate 'getEdges' by division
        divUDmean=(toprow*bottomrow).mean()
        divmeanUD=Tmean*Bmean
              
        # calculate 'getEdges' by substraction 
        frontrow=windowD[:,:,0]
        backrow=windowD[:, :, windowrowvalue-1]
        Fmean=frontrow.mean()
        Bmean=backrow.mean()
        gradmeanFB=Fmean-Bmean
        gradFBmean=(frontrow-backrow).mean()
        
        # calculate 'getEdges' by division
        divFBmean=(frontrow*backrow).mean()
        divmeanFB=Fmean*Bmean
        
        return  gradLRmean, gradmeanLR, divLRmean, divmeanLR, \
                gradUDmean, gradmeanUD, divUDmean, divmeanUD, \
                gradFBmean, gradmeanFB, divFBmean, divmeanFB
    
    
    ############################################################
    # feature[6]= sliceEntropy calculation (disk window or entire image)
    ############################################################
    def getEntropy(self, x, y, z, windowSize):
        #mySlice = self.Data[:,:,z].astype('uint8')
        #entropySlice = entropy(mySlice, disk(windowSize))
        #return entropySlice[x,y]
        return 0
        
    def getEntropyByMask(self, mask3D, windowSize):
        sys.stdout.write("Calculating entropy")
        _,_,d = self.Data.shape
        returnValue = np.array([])
        for z in range(0,d):
            sys.stdout.write('.')
            mySlice = self.Data[:,:,z].astype('uint8')
            #mySlice = img_as_ubyte(mySlice)
            mask = mask3D[:,:,z]
            entropySlice = entropy(mySlice, disk(windowSize))
            result = entropySlice[mask]
            returnValue = np.append(returnValue, result)
        
        print("")
        nbValues = len(returnValue)
        return returnValue.reshape(nbValues, 1)
        #if windowSize not in self.Entropy.keys():
        #    data8 = self.Data.astype('uint8')
        #    self.Entropy[windowSize] = entropy(data8, ball(windowSize))
        
        #return self.Entropy[windowSize][mask3D].T
    
    
    
    def image_entropy(self, z):
        # calculates the sliceEntropy of the entire slice
        img=self.getSlice(z)
        histogram,_ = np.histogram(img,100)
        histogram_length = sum(histogram)
    
        samples_probability = [float(h) / histogram_length for h in histogram]
        image_entropy=-sum([p * math.log(p, 2) for p in samples_probability if p != 0])
    
        return image_entropy
    
    
    ############################################################
    # feature[7]= 3D averaging (Keshani et al)
    ############################################################
    
    def averaging3D (self, x,y,z,windowrowvalue):
               
        # square windowrowvalue x windowrowvalue
        valdown = windowrowvalue // 2
        valup   = valdown + 1
        
        windowDz = self.Data[x-valdown:x+valup,y-valdown:y+valup,z]
        
        #reshape window into array to calculate mean (and variance)
        h,w = windowDz.shape
        arrayD = np.reshape(windowDz, (h*w))
        
        Mz = arrayD.mean()
        
        # nodules will continue in preceeding/succeeding slices but bronchioles will not
        # assume: nodules have minimum length of 5 mm
        Q = int(5 // self.VoxelShape[2] + 1) # = c / T = 5mm / thickness of slices
           
        # mean of same window in preceding slices
        windowDzmin = self.Data[x-valdown:x+valup,y-valdown:y+valup,z-Q:z-1]
        
        h,w,d = windowDzmin.shape
        arrayDmin = windowDzmin.reshape(h*w*d) # make array of 3D matrix
        matrixDmin = arrayDmin.reshape(d, h*w) # make matrix with every row the values of the window per slice
        trans1 = matrixDmin.transpose() # switch rows and columns
        row,_ = trans1.shape
        
        S1 = sum(trans1)/row
        Mpre = S1.mean()
        
        
        # mean of same window in succeeding slices
        windowDzplus = self.Data[x-valdown:x+valup,y-valdown:y+valup,z+1:z+Q]
           
        h,w,d = windowDzplus.shape
        arrayDplus = windowDzplus.reshape(h*w*d) # make array of 3D matrix
        matrixDplus = arrayDplus.reshape(d, h*w) # make matrix with every row the values of the window per slice
        trans2 = matrixDplus.transpose() # switch rows and columns
        row,_ = trans2.shape
        
        S2 = sum(trans2)/row
        Mplus = S2.mean()
        
        #TODO also in other dimensions?
        
        return Mz, Mpre, Mplus # REMARK: Mz is common mean (also in previous function)
    
    
    ############################################################
    # feature[8]= getEdges: sobel
    ############################################################
    
    def getEdges(self, x, y, z): #TODO perform window calculations on these
        #import scipy
        #from scipy import ndimage
        #from scipy.ndimage.filters import generic_gradient_magnitude, sobel
                           
    #     dx = ndimage.sobel(self.Data, 0)  # x derivative
    #     dy = ndimage.sobel(self.Data, 1)  # y derivative
    #     dz = ndimage.sobel(self.Data, 2)  # z derivative
        
        if self.Edges == None:
            self.Edges = generic_gradient_magnitude(self.Data, sobel)
            
        return self.Edges[x,y,z]
    
        #TODO haar-like features
        
