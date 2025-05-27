# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:56:57 2024

@author: komal
"""

import numpy as np

def proj(v, u):
    
    v, u = v[:,np.newaxis], u[:,np.newaxis]
    
    return (v.T@u)/(u.T@u)*u

def Gram_Schmidth(V):
    
    U = np.zeros((V.shape[0],V.shape[1]))
    
    U[:,0][:,np.newaxis] = V[:,0][:,np.newaxis]/np.linalg.norm(V[:,0][:,np.newaxis])

    for i in range(1, U.shape[1]):
        S = 0
        for j in range(i):
            S += proj(V[:,i], U[:,j])
            
        U[:,i][:,np.newaxis] = V[:,i][:,np.newaxis] - S
        U[:,i][:,np.newaxis] = U[:,i][:,np.newaxis]/np.linalg.norm(U[:,i][:,np.newaxis])
        
    return U

# def Gram_Schmidth(V):
    
#     U = np.zeros((V.shape[0],V.shape[1]))
    
#     U[0][:,np.newaxis] = V[0][:,np.newaxis]/np.linalg.norm(V[0][:,np.newaxis])

#     for i in range(1, U.shape[0]):
#         S = 0
#         for j in range(i):
#             S += proj(V[i], U[j])
            
#         U[i][:,np.newaxis] = V[i][:,np.newaxis] - S
#         U[i][:,np.newaxis] = U[i][:,np.newaxis]/np.linalg.norm(U[i][:,np.newaxis])
        
#     return U

def generate_rank_r_matrix(r, no_smpls, q):
    
    mean = np.zeros(r)
    cov = np.eye(r) #[[1,0,0],[0,1,0],[0,0,1]]

    V = np.random.multivariate_normal(mean,cov,no_smpls)
    U = Gram_Schmidth(V)

    B = np.random.multivariate_normal(mean, cov, q)
    X = U@B.T
    
    # return X, U, B
    return X

r = 2
no_smpls = 5
q = 3
##### this function generates a n * q rank r matrix but doesn't work for r = 1 ######## 
# X, U, B = generate_rank_r_matrix(r, no_smpls, q)





    

    