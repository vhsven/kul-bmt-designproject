'''
Created on 9-apr.-2014

@author: Eigenaar
'''
import numpy as np

import pprint
n = 3
distance = [[[0 for k in xrange(n)] for j in xrange(n)] for i in xrange(n)]
pprint.pprint(distance)

distance[1][2][0]=5 #slice, rij, kolom
distance[0][0][0]=3
distance[1][0][0]=4
pprint.pprint(distance)
print(len(distance))




# m,n = a.shape
# b = a.reshape(m*n) # make array of matrix (which sequence?)
# print(b)
# 
# c= b.reshape(m*2,n/2)
# print(c)
# 
# 
# T = c.transpose()
# print(T)
# row,col = T.shape
# print(row)
# print(col)
# 
# S = sum(T)/row
# print(S)
# 
# M = S.mean()
# print(M)

