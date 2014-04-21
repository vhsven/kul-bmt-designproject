import scipy.stats
import numpy as np
import math
import collections
from skimage.filter.rank import entropy
from skimage.morphology import disk
import scipy.ndimage as nd
from scipy.ndimage.filters import generic_gradient_magnitude, sobel

class FeatureGenerator: #TODO fix edge problems
    def __init__(self, data, vshape):
        self.Data = data
        self.VoxelShape = vshape
        self.Edges = None
        self.Blobs = None
        self.PixelCount = None
        
    def getSlice(self, z):
        return self.Data[:,:,int(z)]
    
    ############################################################
    #featurevector[1]= abs/ref position and gray value
    ############################################################
    def getTrivialFeatures(self, x, y, z):
        w,h,d = self.Data.shape

        #return x, y, z, float(x)/w, float(y)/h, float(z)/d, self.Data[x,y,z]

        return float(x)/w, float(y)/h, float(z)/d, self.Data[x,y,z]
    
    
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
        
        # calculate 'getVolumeEdges' by substraction 
        leftrow=windowD[:,0,:]
        rightrow=windowD[:,windowrowvalue-1,:]
        meanL=leftrow.mean()
        meanR=rightrow.mean()
        gradLRmean=(rightrow-leftrow).mean()
        gradmeanLR=meanR-meanL
        
        # calculate 'getVolumeEdges' by division
        divmeanLR=meanR*meanL
        divLRmean=(leftrow*rightrow).mean()
        
        # calculate 'getVolumeEdges' by substraction 
        toprow=windowD[0,:,:]
        bottomrow=windowD[windowrowvalue-1, :, :]
        Tmean=toprow.mean()
        Bmean=bottomrow.mean()
        gradmeanUD=Tmean-Bmean
        gradUDmean=(toprow-bottomrow).mean()
        
        # calculate 'getVolumeEdges' by division
        divUDmean=(toprow*bottomrow).mean()
        divmeanUD=Tmean*Bmean
              
        # calculate 'getVolumeEdges' by substraction 
        frontrow=windowD[:,:,0]
        backrow=windowD[:, :, windowrowvalue-1]
        Fmean=frontrow.mean()
        Bmean=backrow.mean()
        gradmeanFB=Fmean-Bmean
        gradFBmean=(frontrow-backrow).mean()
        
        # calculate 'getVolumeEdges' by division
        divFBmean=(frontrow*backrow).mean()
        divmeanFB=Fmean*Bmean
        
        return  gradLRmean, gradmeanLR, divLRmean, divmeanLR, \
                gradUDmean, gradmeanUD, divUDmean, divmeanUD, \
                gradFBmean, gradmeanFB, divFBmean, divmeanFB
    
    
    ############################################################
    # feature[6]= sliceEntropy calculation (disk window or entire image)
    ############################################################
    def pixelentropy(self, z): #TODO fix this -> astype
        # calculates the sliceEntropy of each pixel in the slice in comparison to its surroundings
        image = self.getSlice(z)
        #pylab.imshow(image, cmap=pylab.gray())
        #pylab.show()
        image = image.view('uint8')
        image = image[:, 1::2]
    #     fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
        
    #     img0 = ax0.imshow(image, cmap=plt.cm.gray)
    #     ax0.set_title('Image')
    #     ax0.axis('off')
    #     plt.colorbar(img0, ax=ax0)
        
        pixelentr=entropy(image, disk(5))
    #     print(imentropy)
    #     img1 = ax1.imshow(imentropy, cmap=plt.cm.jet)
    #     ax1.set_title('Entropy')
    #     ax1.axis('off')
    #     plt.colorbar(img1, ax=ax1)
    #     plt.show()
        return pixelentr #returns a matrix with sliceEntropy values for each pixel
    
    
    
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
    # feature[8]= getVolumeEdges: sobel
    ############################################################
    
    def getVolumeEdges(self): #TODO perform window calculations on these
        #import scipy
        #from scipy import ndimage
        #from scipy.ndimage.filters import generic_gradient_magnitude, sobel
                           
    #     dx = ndimage.sobel(self.Data, 0)  # x derivative
    #     dy = ndimage.sobel(self.Data, 1)  # y derivative
    #     dz = ndimage.sobel(self.Data, 2)  # z derivative
        
        if self.Edges == None:
            self.Edges = generic_gradient_magnitude(self.Data, sobel)
            
        return self.Edges
    
    ############################################################
    # feature[9]= blob detection with laplacian of gaussian
    ############################################################
    def blobdetection(self, z): #TODO do this in 3D, take into account different voxel sizes
        if self.Blobs == None:
            image=self.getSlice(z)
            returnValue = []
            for sigma in np.arange(1.9,2.1,0.1):
                LoG = nd.gaussian_laplace(image, sigma) # scalar: standard deviations of the Gaussian filter
                # hoe bepaal je sigma? nodule_grootte_in_pixels = sqrt(2*sigma)
                # sigma empirisch vastgesteld op 1.9/ 2/ 2.1
                aLoG = abs(LoG)
                output = np.copy(image)
                output[aLoG > aLoG.max()-200] = 1
                #pylab.imshow(output, cmap=pylab.gray())
                #pylab.show()
                returnValue.append(output)
            self.Blobs = returnValue
        
        return self.Blobs
    
    ############################################################  
    #featurevector[10]=haar features
    ############################################################
    
    # problem with import SimpleCV???
        
