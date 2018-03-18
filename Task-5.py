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
tol = 3
A = np.random.rand(k,n) #n basis vector
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
signs.shape = (signs.shape[0],1)
flag1 = 0
flag2 = 0
flag_first = 1
itr =0
while((flag1 != 1 or flag2!= 1)and (itr <50)):
    itr +=1
    print("itr",itr)
#step 2
    if((flag1 == 1 and flag2 != 1) or flag_first == 1):
        flag_first = 0
        for i,idx in enumerate(zero_coeff):
            temp = A[:,idx]
            temp.shape = (k,1)
            prod = np.matmul(A,x)
            prod.shape = (k,1)
            res = abs(-2*np.matmul((y-prod).T,temp))
            print(res)
            if (res > max_res[0]):
                max_res = (res,idx)
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
#step 3
    #Transpose may not work in following.
    if(len(A_hat.shape) != 2):
        A_hat.shape = (A_hat.shape[0],1)
    if(len(signs_hat.shape) != 2):
        signs_hat.shape = (signs_hat.shape[0],1)
    if(len(x_hat.shape) != 2):
        x_hat.shape = (x_hat.shape[0],1)
    #A_dash_A = np.matmul(A_hat.T,A_hat)
#    x_new = np.matmul(np.linalg.pinv(np.matmul(A_hat.T,A_hat)), (np.matmul(A_hat.T,y)-(gamma/2)*signs_hat))
    x_new = np.linalg.solve(np.atleast_2d(np.matmul(A_hat.T,A_hat)), np.matmul(A_hat.T,y)-(gamma/2)*signs_hat)
    print("x_new=",x_new)
    sign_change = []
    for i,x_h_i in enumerate(x_hat):
        if((x_h_i >= 0 and x_new[i] < 0) or (x_h_i <= 0 and x_new[i] > 0)):
            sign_change.append(i)
    # create new x with the values in x_hat replaced. check for min and update
    x_temp = x
    for i,idx in enumerate(active_set):
        x_temp[idx] = x_new[i]
    x_min = np.array([0]*n)
    zeroed_idx = -1
    min_obj_f = pow(np.linalg.norm(y-np.matmul(A,x_temp)),2) + gamma*np.linalg.norm(x_temp, 1)
    for i,idx in enumerate(sign_change):
        const = -x_hat[idx]/(x_new[idx] - x_hat[idx])
        x_h_t = x_hat + const*(x_new - x_hat)
        x_temp = x
        for j,jdx in enumerate(active_set):
            x_temp[jdx] = x_h_t[j]
        obj_f =  pow(np.linalg.norm(y-np.matmul(A,x_temp)),2) + gamma*np.linalg.norm(x_temp, 1)
        print("obj_f is=",obj_f," for k =",const ,"min_obj_f=",min_obj_f)
        if(obj_f < min_obj_f):
            x_min = x_temp
            zeroed_idx = idx
    x = x_min
    for i in range(n):
        if(x[i]>0):
            signs[i] = 1
        elif(x[i]<0):
            signs[i] = -1
        elif(x[i] == 0):
            signs[i] =0
    if(zeroed_idx != -1):
        print("zeroed_idx = ",zeroed_idx)
        x_hat = np.delete(x_hat,zeroed_idx)
        signs[active_set[zeroed_idx]] = 0
        zero_coeff.append(active_set[zeroed_idx])
        del active_set[zeroed_idx]
        signs_hat = np.delete(signs_hat,zeroed_idx)
        A_hat = np.delete(A_hat,zeroed_idx, 1)
    flag1 =1
    for i, idx in enumerate(active_set):
        temp = A[:,idx]
        #print(temp)
        temp.shape = (k,1)
        prod = np.matmul(A,x)
        prod.shape = (k,1)
        res = -2*np.matmul((y-prod).T,temp) +gamma*signs[idx]
        if(res < -tol or res > tol): #i.e if unsatisfied
            flag1 = 0
    flag2 = 1
    for i,idx in enumerate(zero_coeff):
        temp = A[:,idx]
        temp.shape = (k,1)
        prod = np.matmul(A,x)
        prod.shape = (k,1)
        res = abs(-2*np.matmul((y-prod).T,temp))
        if(res > gamma):
            flag2 = 0
    print(x)
