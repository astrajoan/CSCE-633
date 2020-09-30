# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:20:58 2020

@author: yvonn
"""
import numpy as np
x = np.array([[1,3],
              [2,4],
              [1,1]])
y = np.array([[1,2],
     [3,4],
     [5,6]])
z = y[0:0,:]

#print(np.concatenate((y,z),axis=0))
a1 = np.array([[1] for i in range(10)])

ax = np.array([[1,2],
      [3,4]])
ay = np.array([[2],
      [3]])
az = np.array([2,3])
#print(np.matmul(x,az))

class abc(object):
    def __init__(self,a):
        self.a = a
    def add(self,b):
        self.a = self.a+b
    def subtr(self,c):
        self.a = self.a-c
    def prints(self):
        print(self.a)
        
        
n = abc(1)
n.add(2)
n.prints()
n.add(5)
n.prints()
n.subtr(2)
n.prints()

u = np.array([1,2,3])
v = np.array([2,3,4])
c = np.array([3,4,5]).reshape(3,1)
uv = np.vstack((u,v)).transpose()
uv = np.concatenate((uv,c),axis=1)
print(uv)
