# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:35:28 2024

@author: Komal
"""

import numpy as np
from LRMC import LowRankMatrixRecovery
from simulated_ORL_data import OfflineData

class PolicyEvaluation:
    def __init__(self):
        self.Q = []
        self.Z = []
        self.Y = []

    def STKE(n_s, n_a, data, t, K):
        n_t = np.zeros((n_s,n_a))
        emp_P = np.zeros((n_s,n_a,n_s))
        r = np.zeros((n_s,n_a))
        for k in range(K):
            s_b, a_b = data[k][t][0], data[k][t][1][0]
            n_t[s_b,a_b] += 1
        for k in range(K):
            s_b, a_b = data[k][t][0], data[k][t][1][0]
            r[s_b,a_b] = data[k][t][3]
            s1_b = data[k][t][2]
            # n_t[s_b,a_b] += 1
            emp_P[s_b,a_b,s1_b] += 1/n_t[s_b,a_b]
        return n_t, n_t/K, emp_P, r


    def Y_calculation(T_P_Q, beh_SAOM, R, Y):
        
        support = np.nonzero(beh_SAOM)
        for i in range(len(support[0])):
            s, a  = support[0][i], support[1][i]
            Y[s,a] = R[s,a] + np.sum(T_P_Q)
            
        return Y

    def policy_evaluation(self, beh_data, beh_pol, m, n_s, n_a, P, R):
        
        H = len(beh_data[0])
        K = len(beh_data)
        
        self.Q.append(np.zeros((n_s,n_a)))
        
        error = 0
        
        for i in range(H):
            self.Z.append(np.zeros((n_s,n_a)))
            self.Y.append(np.zeros((n_s,n_a)))
            self.Q.append(np.zeros((n_s,n_a)))
        
        for t in reversed(range(H)):
            
            tar_SAOM = (1/(m*n_s))*np.random.binomial(1, m/n_s, (n_s,n_a))

            beh_SAOM = (1/n_s)*beh_pol[t]
        
            n_t, rho_t, P_t_est, r_t = self.STKE(n_s, n_a, beh_data, t, K)
            
            ######## Q iteration ########
            beh_support = np.nonzero(n_t)
            
            P_Q   = (n_s*tar_SAOM)*self.Q[t+1]
            T_P_Q = tar_SAOM*self.Q[t+1]
            
            for i in range(len(beh_support[0])):
                s, a = beh_support[0][i], beh_support[1][i]
                
                sum_term = np.sum((P_t_est[s,a,:][:,np.newaxis]@np.ones((1,n_s)))*P_Q)
                        
                self.Z[t][s,a] = r_t[s,a] + sum_term
                
            self.Y[t] = self.Y_calculation(T_P_Q, beh_SAOM, R, self.Y[t])
                
            self.Q[t] = LowRankMatrixRecovery.AltGDmin(self.Z[t], rho_t)
            # Q[t] = LowRankMatrixRecovery.Altmin(self.Z[t], rho_t)
            
            tar_support = np.nonzero(tar_SAOM)

            for i in range(len(tar_support[0])):
                s, a = tar_support[0][i], tar_support[1][i]
                error += tar_SAOM[s,a]*(self.Q[t][s,a] - self.Y[t][s,a])
            
        return error


def main(n_s, n_a, h, m, K):
    
    avg_error = 0
    for _ in range(5):
        
        ########## data generation ##########
        beh_data, P, R, beh_pol = OfflineData.data_gen(K, m, n_s, n_a, h)
        ########## policy evaluation ########
        error = PolicyEvaluation.policy_evaluation(beh_data, beh_pol, m, n_s, n_a, P, R)
    
        avg_error += np.abs(error)
         
    return avg_error/5
    
# n_s = 20
# n_a = 20
# # H   = [5, 10, 15]
# h   = 5
# K   = 500
# # M   = [10, 50, 100]
# m = 1
    
    
    
    
    
    