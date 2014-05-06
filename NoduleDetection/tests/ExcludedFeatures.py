import numpy as np
import collections
import sys
from skimage.morphology.selem import disk
from skimage.filter.rank.generic import entropy
import math

class ExcludedFeatures:
    def __init__(self):
        self.Data = []
        
    def getRelativePosition(self, x, y, z):
        h,w,d = self.Data.shape
        return float(x)/h, float(y)/w, float(z)/d
     
    def getRelativePositionByMask(self, mask3D):
        h,w,d = self.Data.shape
        xs, ys, zs = np.where(mask3D)
        xsr = xs / float(h)
        ysr = ys / float(w)
        zsr = zs / float(d)
        return np.vstack([xsr,ysr,zsr]).T
    
    def neighbours(self, x,y,z):
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
        
        return  greyvalue,M,V,\
                Max_greyvalue,Min_greyvalue,maxdiff,mindiff,maxdiv,minplus,maxplus, mindiv,maxmindiff,\
                freq_pixelvalue,freq_max,freq_min #cx,cy,cz,sx,sy,sz     
                
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