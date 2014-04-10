'''
Created on 9-apr.-2014

@author: Eigenaar
'''
import numpy as np
from numpy import linalg

data=np.ones((4,4))
data=data*5
data[1,1]=6
x=1
y=1

data=np.ones((5,5))
data=data*5
data[1,4]=6
data[2,2]=7
print (data)
distL=data[:,0]
distR=data[:,(5-1)]
dist=distL-distR
print(dist)