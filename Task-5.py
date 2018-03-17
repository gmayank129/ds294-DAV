#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 20:11:31 2018

@author: mayank
"""

import numpy as np
#from scipy.stats import ortho_group
n = 8 #10 basis vectors
k = 15 #dimension of input vectors
x = np.array([0]*n) #sparse weight vectors
y = np.random.rand(k,1) #input
gamma = 2.0 #constant random value

A = np.random.rand(k,n)
zero_coeff = [] #list having index of coefficients with zero values
signs = np.array([])      #sign of all x
active_set=[]   #list having index of coefficients with non-zero values
A_hat = np.array([])  #Basis vectors with non_zero coeffcient
x_hat = np.array([])  #Array of non_zero coeffcient
signs_hat = np.array([])  #Array of signs of non_zero coeffcient
max_res = (0,-1)

for i in range(0,n):
    zero_coeff.append(i)
    signs = np.append(signs,0)
    
for idx,i in enumerate(zero_coeff):
    temp = A[:,i]
    temp.shape = (k,1)
    prod = np.matmul(A,x)
    prod.shape = (k,1)
    res = abs(-2*np.matmul((y-prod).T,temp))
    print(res)
    if (res > max_res[0]):
        max_res = (res,i)
print(max_res)
if( max_res[0] > gamma):
    signs[max_res[1]] = -1
    zero_coeff.remove(max_res[1])
    active_set.append(max_res[1])
    A_hat = np.append(A_hat, A[:,max_res[1]])
    x_hat = np.append(x_hat, x[max_res[1]])
    signs_hat = np.append(signs_hat, signs[max_res[1]])
elif(2*np.dot((y-np.matmul(A,x)),A[:,max_res[1]]) < -gamma):
    signs[max_res[1]] = 1
    zero_coeff.remove(max_res[1])
    active_set.append(max_res[1])
    A_hat = np.append(A_hat, A[:,max_res[1]])
    x_hat = np.append(x_hat, x[max_res[1]])
    signs_hat = np.append(signs_hat, signs[max_res[1]])

#Transpose may not work in following.
A_dash_A = np.matmul(A_hat.T,A_hat)

x_new = np.matmul(np.linalg.inv(np.matmul(A_hat.T,A_hat)), (np.matmul(A_hat.T,y)-(gamma/2)*signs))
sign_change = []
for i,x_h_i in enumerate(x_hat):
    if((x_h_i > 0 and x_new[i] < 0) or (x_h_i < 0 and x_new[i] > 0)):
        sign_change.append(i)


for idx,i in enumerate(sign_change):
    k = x_hat[i]
    
    