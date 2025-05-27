# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:23:09 2024

@author: komal
"""

import numpy as np
from matplotlib import pyplot as plt

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

def generate_rank_r_matrix(r, no_smpls, q):

    mean = np.zeros(r)
    cov = np.eye(r) #[[1,0,0],[0,1,0],[0,0,1]]

    V = np.random.multivariate_normal(mean,cov,no_smpls)
    U = Gram_Schmidth(V)

    B = np.random.multivariate_normal(mean, cov, q)
    X = U@B.T

    return X, U, B


n = 3000
q = 5000

r = 10
p = 0.05

def low_rank_observations(n, q, r, p, sigma):

    X, U, B = generate_rank_r_matrix(r, n, q)

    M = np.random.binomial(1, p, (n,q))

    N = sigma*np.random.randn(X.shape[0],X.shape[1])

    Y = M * X + M * N

    return Y, M, U

def AltGDmin(Y, M, U, r, T, p):

    U_y, S_y, V_y = np.linalg.svd(Y, full_matrices=False)
    
    Y_op      = S_y[0]
    step_size = 0.1*p/Y_op**2
    
    U_0 = U_y[:,:r]
    n   = Y.shape[0]
    B   = np.zeros((r,q))
    SD  = [np.linalg.norm((np.eye(n) - U_0 @ U_0.T) @ U, 'fro')]

    for t in range(T):
        for k in range(q):
            m_k    = np.nonzero(M[:,k])[0]
            U_k    = U_0[m_k,:]
            B[:,k] = np.linalg.inv(U_k.T@U_k) @ U_k.T @ Y[m_k,k]

        grad_U = 2 * (M * (U_0@B) - Y) @ B.T
        U_1    = U_0 - step_size * grad_U
        U_1, _ = np.linalg.qr(U_1)

        U_0 = U_1

        SD.append(np.linalg.norm((np.eye(n) - U_0 @ U_0.T) @ U, 'fro'))
        
    return SD


######## U_0 initialization ######
# mu = 0.7
# mu_1 = mu*np.sqrt(r/n)

# M_0 = np.zeros((U_00.shape[0],U_00.shape[1]))
# for i in range(U_00.shape[0]):
#     # print(min(1,mu_1/np.linalg.norm(U_00[i])))
#     M_0[i] = U_00[i]*min(1,mu_1/np.linalg.norm(U_00[i]))

# U_0, _ = np.linalg.qr(M_0)

T = 300
# rank = [5, 10, 20]
std_dev = [1e-12, 1e-6, 1e-3]


plt.figure()
plt.yscale('log')
for sigma in std_dev:
    
    Y, M, U = low_rank_observations(n, q, r, p, sigma)
    SD = AltGDmin(Y, M, U, r, T, p)
    # plt.plot(range(0,T+1,100), SD[::100], marker='x', markersize=10)
    plt.plot(range(T+1), SD, label='$\sigma =$ ' + str(sigma))
    
plt.grid()
plt.ylabel(r'SD$(U^{(t)},U)$')
plt.xlabel('iterations')
plt.title('Noisy LRMC performance')
plt.legend()
plt.savefig("(%d,%d) Noisy LRMC with %d observations and rank %d.png" % (n, q, p*n*q, r))
plt.show()


# plt.figure()
# plt.yscale('log')
# for r in rank:
    
#     Y, M, U = low_rank_observations(n, q, r, p)
#     SD = AltGDmin(Y, M, U, r, T, p)
#     # plt.plot(range(0,T+1,100), SD[::100], marker='x', markersize=10)
#     plt.plot(range(T+1), SD, label='rank ' + str(r))
    
# plt.grid()
# plt.ylabel(r'SD$(U^{(t)},U)$')
# plt.xlabel('iterations')
# plt.title('Noisy LRMC performance')
# plt.legend()
# plt.savefig("(%d,%d) Noisy LRMC with %d observations.png" % (n, q, p*n*q))
# plt.show()

# P = [0.02, 0.05, 0.1]
# T = 700
# plt.figure()
# plt.yscale('log')
# for p in P:
    
#     Y, M, U = low_rank_observations(n, q, r, p)
#     SD = AltGDmin(Y, M, U, r, T, p)
#     # plt.plot(range(0,T+1,100), SD[::100], marker='x', markersize=10)
#     plt.plot(range(T+1), SD, label=str(p*100)+'% of observations')
    
# plt.grid()
# plt.ylabel(r'SD$(U^{(t)},U)$')
# plt.xlabel('iterations')
# plt.title('Noise free LRMC performance')
# plt.legend()
# plt.savefig("(%d,%d) LRMC with rank %d.png" % (n, q, r))
# plt.show()



# print(SD)

# fig, ax = plt.subplots()

# ax.set_yscale('log')
# # ax.semilogy(range(T+1), SD)
# ax.plot(range(T+1), SD)
# ax.grid()
# ax.set_ylabel(r'SD$(U^{(t)},U)$')
# ax.set_xlabel('iterations')
# ax.figure.savefig('LRMC with rank {}'.format(r))
# plt.show()



