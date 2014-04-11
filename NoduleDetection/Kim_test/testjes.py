import numpy as np
import matplotlib.pyplot as plt
import time

plt.figure(figsize=(10,10))

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax4 = plt.subplot(224)

#Make some sample data as a sum of two elliptical gaussians:
x = range(200)
y = range(200)
print(x)
print(y)

X,Y = np.meshgrid(x,y)
print(X)
print(Y)

def twoD_gaussian(X,Y,A=1,xo=100,yo=100,sx=20,sy=10):
    return A*np.exp(-(X-xo)**2/(2.*sx**2)-(Y-yo)**2/(2.*sy**2))

Z = twoD_gaussian(X,Y) + twoD_gaussian(X,Y,A=0.4,yo=75)
print(Z)

ax2.imshow(Z) #plot it

h,w = np.shape(Z)


#calculate projections along the x and y axes for the plots
yp = np.sum(Z,axis=1)
xp = np.sum(Z,axis=0)
print(yp)
print(xp)


windowD = np.ones((5,5))

windowD = windowD*2
windowD[1,2]=6
windowD[2,2]=6

print (windowD)
h,w,=windowD.shape
arrayD = np.reshape(windowD, (h*w*d))
print(arrayD)

# mean and variance
M=arrayD.mean()
V=arrayD.var()

print(M)
print(V)

rangex = range(w)
rangey = range(h)
rangez = range(d)

print(rangex)
print(rangey)
print(rangez)


#calculate projections along the x and y axes
zp = np.sum(windowD,axis=2)
yp = np.sum(windowD,axis=1)
xp = np.sum(windowD,axis=0)

print(zp)
print(yp)
print(xp)
#centroid
cx = np.sum(rangex*xp)/np.sum(xp)
cy = np.sum(rangey*yp)/np.sum(yp)
cz = np.sum(rangez*zp)/np.sum(zp)

#standard deviation
x2 = (rangex-cx)**2
y2 = (rangey-cy)**2
z2 = (rangez-cz)**2

sx = np.sqrt( np.sum(x2*xp)/np.sum(xp) )
sy = np.sqrt( np.sum(y2*yp)/np.sum(yp) )
sz = np.sqrt( np.sum(z2*zp)/np.sum(zp) )

#skewness
x3 = (rangex-cx)**3
y3 = (rangey-cy)**3
z3 = (rangez-cz)**3

skx = np.sum(xp*x3)/(np.sum(xp) * sx**3)
sky = np.sum(yp*y3)/(np.sum(yp) * sy**3)
skz = np.sum(zp*z3)/(np.sum(zp) * sz**3)

#Kurtosis
x4 = (rangex-cx)**4
y4 = (rangey-cy)**4
z4 = (rangez-cz)**4
kx = np.sum(xp*x4)/(np.sum(xp) * sx**4)
ky = np.sum(yp*y4)/(np.sum(yp) * sy**4)
kz = np.sum(zp*z4)/(np.sum(zp) * sz**4)

#autocorrelation
result = np.correlate(arrayD, arrayD, mode='full')
autocorr=result[result.size/2:]