# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:35:28 2024

@author: Komal
"""

import numpy as np
from MC_for_ORL import AltGDmin
from Seq_simulated_ORL_Data import OfflineData

class PolicyEvaluation:
    def __init__(self):
        self.Q = []
        self.Z = []
        self.Z_hat = []
        # self.Y = []
        self.Q_hat = []

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
            emp_P[s_b,a_b,s1_b] += 1/n_t[s_b,a_b]
        return n_t, n_t/K, emp_P, r


    # def Y_calculation(T_P_Q, beh_SAOM, R, Y):
        
    #     support = np.nonzero(beh_SAOM)
    #     for i in range(len(support[0])):
    #         s, a  = support[0][i], support[1][i]
    #         Y[s,a] = R[s,a] + np.sum(T_P_Q)
            
    #     return Y

    def policy_evaluation(self, beh_data, m, n_s, n_a):
        
        H = len(beh_data[0])
        K = len(beh_data)
        
        self.Q.append(np.zeros((n_s,n_a)))
        self.Q_hat.append(np.zeros((n_s, n_a)))
        
        error = 0
        
        for i in range(H):
            self.Z.append(np.zeros((n_s,n_a)))
            # self.Y.append(np.zeros((n_s,n_a)))
            self.Q.append(np.zeros((n_s,n_a)))
            self.Q_hat.append(np.zeros((n_s,n_a)))
            self.Z_hat.append(np.zeros((n_s,n_a)))
        
        for t in reversed(range(H)):
            
            tar_SAOM = (1/(m*n_s))*np.random.binomial(1, m/n_s, (n_s,n_a))

            # beh_SAOM = (1/n_s)*beh_pol[t]
        
            n_t, rho_t, P_t_est, r_t = PolicyEvaluation.STKE(n_s, n_a, beh_data, t, K)
            
            ######## Q iteration ########
            beh_support = np.nonzero(n_t)
            
            P_Q_hat   = (n_s*tar_SAOM)*self.Q_hat[t+1]
            T_P_Q = (tar_SAOM)*self.Q[t+1]
            # T_P_Q = tar_SAOM*self.Q[t+1]
            sum_term = np.sum(T_P_Q)

            # print("state transition kernel", P_t_est)
            
            for i in range(len(beh_support[0])):
                s, a = beh_support[0][i], beh_support[1][i]
                
                sum_term_hat = np.sum((P_t_est[s,a,:][:,np.newaxis]@np.ones((1,n_a)))*P_Q_hat)
                        
                # self.Z[t][s,a] = r_t[s,a] + sum_term

                self.Z_hat[t][s,a] = r_t[s,a] + sum_term_hat
                self.Z[t][s,a] = r_t[s,a] + sum_term
                
            # self.Y[t] = PolicyEvaluation.Y_calculation(T_P_Q, beh_SAOM, R[t], self.Y[t])
            # if m != n_s:
            self.Q[t] = AltGDmin(self.Z[t], rho_t) # noise-free AltGDmin
            self.Q_hat[t] = AltGDmin(self.Z_hat[t], rho_t) # noisy AltGDmin
            # else:
            #     self.Q[t] = self.Z[t]
            #     self.Q_hat[t] = self.Z_hat[t]

            # self.Q[t] = self.Z[t]
            # Q[t] = Altmin(self.Z[t], rho_t)
            
            # tar_support = np.nonzero(tar_SAOM)

            # for i in range(len(tar_support[0])):
            #     s, a = tar_support[0][i], tar_support[1][i]
            #     error += tar_SAOM[s,a]*(self.Q[t][s,a] - self.Y[t][s,a])
        
            # print("true expected state-action value on support", self.Z[t])
            # print("estimated expected state-action value on support", self.Z_hat[t])
            # print("after estimating out of support from true values", self.Q[t])
            # print("after estimating out of support from noisy values", self.Q_hat[t])

        dev = self.Q[t] - self.Q_hat[t]
        error = np.sum(dev*dev)
        true_Q_norm = np.sum(self.Q[t]*self.Q[t])
        norm_error = error/true_Q_norm

        return error, norm_error


def main(n_s, n_a, h, m, K, rank, reward_level, no_trials):
    
    avg_error = 0
    avg_norm_error = 0
    print("-------------------------------------------------------------------------------")
    print("Number of States %d, Action subset sizes %d, Horizon %d and Trajectories %d"%(n_s, m, h, K))
    
    for i in range(no_trials):
        print("Trial ", (i+1))
        ########## data generation ##########
        Data_Generation = OfflineData(n_s, n_a, m, h, K, rank, reward_level)
        beh_data = Data_Generation.data_gen()
        ########## policy evaluation ########
        evaluate_policy = PolicyEvaluation()
        error, norm_error = evaluate_policy.policy_evaluation(beh_data, m, n_s, n_a)
        print("Error and norm error are", (error, norm_error))
        avg_error += np.abs(error)
        avg_norm_error += np.abs(norm_error)
         
    return avg_error/no_trials, avg_norm_error/no_trials

if __name__ == "__main__":
    n_s = 100
    n_a = 30
    # H   = [5, 10, 15]
    h   = 5
    K   = 1000000
    # M   = [10, 50, 100]
    A_s = 15
    no_trials = 5
    rank = 1
    reward_level = 1
    error, norm_error = main(n_s, n_a, h, A_s, K, rank, reward_level, no_trials)
    print("Average error and norm error are", error, norm_error)
    
    
    
    
    