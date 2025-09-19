# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:23:09 2024

@author: Komal
"""

import numpy as np
from matplotlib import pyplot as plt
from Gram_Schmidth import LinearAlgebra

class LowRankMatrixRecovery:

    def low_rank_observations(self, n, q, r, p, sigma):

        X, U, B = LinearAlgebra.generate_rank_r_matrix(r, n, q)

        M = np.random.binomial(1, p, (n,q))

        N = sigma*np.random.randn(X.shape[0],X.shape[1])

        Y = M * X + M * N

        return Y, M, U, X

    def LS(self, S, A, Z):

        X = np.zeros((A.shape[1],Z.shape[1]))
        for k in range(Z.shape[1]):
            s_k    = np.nonzero(S[:,k])[0]
            X[:,k] = np.linalg.inv(A[s_k,:].T@A[s_k,:])@A[s_k,:].T@Z[s_k,k]
            # X[:,k] = np.linalg.lstsq(A[s_k,:],Z[s_k,k])[0]

        return X

    # def U0_init(A):
        
    #     x_k = np.random.rand(A.shape[1])

    #     for k in range(T):

    #         x_k1 = A @ x_k
    #         x_k1_norm = np.linalg.norm(x_k1)
    #         x_k = x_k1/x_k1_norm

    #     return x_k


    def AltGDmin(self, X, Y, M, U, r, T, p):

        U_y, S_y, V_y = np.linalg.svd(Y, full_matrices=False)
        
        Y_op      = S_y[0]
        step_size = 0.1*p/Y_op**2
        
        U_0 = U_y[:,:r]
        n   = Y.shape[0]
        B   = np.zeros((r,q))
        SD  = [np.linalg.norm((np.eye(n) - U_0 @ U_0.T) @ U, 'fro')]
        error = []

        for t in range(T):
            # for k in range(q):
                # m_k    = np.nonzero(M[:,k])[0]
                # U_k    = U_0[m_k,:]
                # B[:,k] = np.linalg.inv(U_k.T@U_k) @ U_k.T @ Y[m_k,k]
            B = self.LS(M, U_0, Y)

            grad_U = 2 * (M * (U_0@B) - Y) @ B.T
            U_1    = U_0 - step_size * grad_U
            U_1, _ = np.linalg.qr(U_1)

            U_0 = U_1

            SD.append(np.linalg.norm((np.eye(n) - U_0 @ U_0.T) @ U, 'fro'))
        error = np.linalg.norm(X-U_0@B, 'fro')/np.linalg.norm(X,'fro')
        print(error)
            
        return SD

    # def sample_split(M, T):

    #     Ms = [np.zeros(M.shape) for t in range(2*T+1)]

    #     for i in range(M.shape[0]):
    #       for j in range(M.shape[1]):
    #         idx = np.random.randint(0, 2*T)
    #         Ms[idx][i,j] = M[i,j]

    #     return Ms


    def Altmin(self, X, Y, M, U, r, T, p):

        U_y, S_y, V_y = np.linalg.svd(Y, full_matrices=False)

        U_0 = U_y[:,:r]

        n   = Y.shape[0]
        B   = np.zeros((r,q))
        SD  = [np.linalg.norm((np.eye(n) - U_0 @ U_0.T) @ U, 'fro')]

        for t in range(T):

            B      = self.LS(M, U_0, Y)

            U_1    = self.LS(M.T, B.T, Y.T).T
        
            U_1, _ = np.linalg.qr(U_1)

            U_0    = U_1

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



def main():

    n = 3000
    q = 5000

    R = [5, 10]
    P = [0.05, 0.1] 
    # sigma = 0
    T = 600
    # rank = [10, 20, 5]
    std_dev = [1e-1, 1e-3, 1e-6, 1e-12]
    # std_dev = [0]

    for r in R:
        for p in P:

            print(r)
            print(p)
            
            plt.figure()
            plt.yscale('log')
            for sigma in std_dev:
                
                Y, M, U, X = LowRankMatrixRecovery.low_rank_observations(n, q, r, p, sigma)
                SD = LowRankMatrixRecovery.AltGDmin(X, Y, M, U, r, T, p)
                # SD = LowRankMatrixRecovery.Altmin(X, Y, M, U, r, T, p)
                # plt.plot(range(0,T+1,100), SD[::100], marker='x', markersize=10)
                np.savez("./LRMC results/Noisy (%d, %d) LRMC for rank %d, probability %.2f, and sigma %s with AltGDmin" %(n, q, r, p, str(sigma)), SD)
                plt.plot(range(T+1), SD, label='$\sigma = $' + str(sigma))
                
            plt.grid()
            plt.ylabel(r'SD$(U^{(t)},U)$')
            plt.xlabel('iterations')
            plt.title('rank $r = $ %d and probability $p = $ %.2f' %(r, p))
            plt.legend()
            plt.savefig("./LRMC results/Noisy (%d,%d) AltGDmin LRMC with rank %d and %d observations.png" % (n, q, r, p*n*q))
            plt.show()

    # plt.figure()
    # plt.yscale('log')
    # for r in rank:
        
    #     Y, M, U, X = low_rank_observations(n, q, r, p, sigma)
    #     # SD = AltGDmin(X, Y, M, U, r, T, p)
    #     SD = Altmin(X, Y, M, U, r, T, p)
    #     # plt.plot(range(0,T+1,100), SD[::100], marker='x', markersize=10)
    #     plt.plot(range(T+1), SD, label='rank ' + str(r))
        
    # plt.grid()
    # plt.ylabel(r'SD$(U^{(t)},U)$')
    # plt.xlabel('iterations')
    # plt.title('Noise free LRMC performance by Altmin')
    # plt.legend()
    # plt.savefig("(%d,%d) Noise free Altmin LRMC with %d observations.png" % (n, q, p*n*q))
    # plt.show()

    # P = [0.02, 0.05, 0.1]
    # T = 100
    # plt.figure()
    # plt.yscale('log')
    # for p in P:
        
    #     Y, M, U, X = low_rank_observations(n, q, r, p, sigma)
    #     # SD = AltGDmin(X, Y, M, U, r, T, p)
    #     SD = Altmin(X, Y, M, U, r, T, p)
    #     # plt.plot(range(0,T+1,100), SD[::100], marker='x', markersize=10)
    #     plt.plot(range(T+1), SD, label=str(p*100)+'% of observations')
        
    # plt.grid()
    # plt.ylabel(r'SD$(U^{(t)},U)$')
    # plt.xlabel('iterations')
    # plt.title('Noise free LRMC performance by Altmin')
    # plt.legend()
    # plt.savefig("(%d,%d) Noise free Altmin LRMC with rank %d.png" % (n, q, r))
    # plt.show()

if __name__ == '__main__':
    main()


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



