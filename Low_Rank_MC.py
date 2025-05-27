# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 19:26:04 2024

@author: komal
"""

import numpy as np

from Gram_Schmidth import *
from matplotlib import pyplot as plt

n = 3000
q = 5000

r = 5
p = 0.05

def low_rank_observations(n, q, r, p):

    X, U, B = generate_rank_r_matrix(r, n, q)
    
    M = np.random.binomial(1, p, (n,q))
    
    Y = M * X
    
    return Y, M, U


Y, M, U = low_rank_observations(n, q, r, p)

U_y, S_y, V_y = np.linalg.svd(Y, full_matrices=False)

Y_op = S_y[0]
step_size = 0.1*p/Y_op**2
T = 700

######## U_0 initialization ######
mu = 0.7
mu_1 = mu*np.sqrt(r/n)
U_00 = U_y[:,:r]
M_0 = np.zeros((U_00.shape[0],U_00.shape[1]))
for i in range(U_00.shape[0]):
    M_0[i] = U_00[i]*min(1,mu_1/np.linalg.norm(U_00[i]))

U_0, _ = np.linalg.qr(M_0)

B = np.zeros((r,q))

SD = [np.linalg.norm((np.eye(n) - U_0 @ U_0.T) @ U, 'fro')]

for t in range(T):
    for k in range(q):
        m_k    = np.nonzero(M[:,k])[0]
        U_k    = U_0[m_k,:]
        B[:,k] = np.linalg.inv(U_k.T@U_k) @ U_k.T @ Y[m_k,k]
    
    grad_U = 2 * (M * (U_0@B) - Y) @ B.T
    U_1 = U_0 - step_size * grad_U
    U_1, _ = np.linalg.qr(U_1)
    
    U_0 = U_1
    # print(U_0@U_0.T)
    SD.append(np.linalg.norm((np.eye(n) - U_0 @ U_0.T) @ U, 'fro'))

print(SD)    
fig, ax = plt.subplots()

ax.semilogy(range(T+1), SD)
ax.grid()

plt.show()

