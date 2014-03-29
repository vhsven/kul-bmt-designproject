'''
Created on 28-mrt.-2014

@author: Eigenaar
'''
from scipy import ndimage
import numpy as np
import pylab
square = np.zeros((32, 32))
square[10:-10, 10:-10] = 1
np.random.seed(2)
x, y = (32*np.random.random((2, 20))).astype(np.int)
square[x, y] = 1
open_square = ndimage.binary_opening(square)
pylab.imshow(open_square, cmap=pylab.gray())
pylab.show()
eroded_square = ndimage.binary_erosion(square)
pylab.imshow(eroded_square, cmap=pylab.gray())
pylab.show()
reconstruction = ndimage.binary_propagation(eroded_square, mask=square)
pylab.imshow(reconstruction, cmap=pylab.gray())
pylab.show()