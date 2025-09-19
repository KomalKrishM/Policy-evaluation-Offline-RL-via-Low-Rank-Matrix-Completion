# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:35:23 2024

@author: Komal
"""

import numpy as np
import random

class OfflineData:
    def MDP(n_s, n_a, m, H):

        states = np.arange(n_s)
        actions = np.arange(n_a)
        
        InStDi = np.ones(n_s)/n_s
        
        reward_level = 1

        d = 1
        R = reward_level*np.random.rand(n_s,d)@np.ones((n_a,d)).T
        P = np.ones((n_s,n_a,n_s))/n_s

        # reward = []
        # P = []
        # for _ in range(H):
        #     ######## generating rank-1 reward and transition tensor ########
        #     # r_t = np.random.rand(n_s,1)@np.ones((n_a,1)).T
            
        #     reward.append(r_t * reward_level)
        #     P.append(P_t)

        beh_policy = []
        # tar_policy = []
        for _ in range(H):
                
            beh_policy.append((1/m)*np.random.binomial(1, m/n_s, (n_s,n_a)))
            # tar_policy.append((1/m)*np.random.binomial(1, m/n_s, (n_s,n_a)))
            
        # return states, actions, P, R, InStDi, beh_policy, tar_policy
        return states, actions, P, R, InStDi, beh_policy

    def data_sample(states, P, reward, pol, InSt):
        
        actions_available = np.nonzero(pol[InSt,:])[0]
        action = random.sample(list(actions_available), 1)
        NeSt   = np.random.choice(states, p=P[InSt,action[0]])
        rew    = reward[InSt,action[0]]

        return action, NeSt, rew

    def data_gen(self, K, m, n_s, n_a, H):
        
        ###### generate a MDP ######
        # states, actions, P, reward, InStDi, beh_pol, tar_pol = MDP(n_s, n_a, m, H)
        states, _, P, reward, InStDi, beh_pol = self.MDP(n_s, n_a, m, H)
        
        ###### sample K trajectories from generated MDP ####### 
        beh_data = []
        # tar_data = []
        for _ in range(K):
            beh_trajectory = []
            # tar_trajectory = []
            # tar_InSt = np.random.choice(states, p=InStDi)
            beh_InSt = np.random.choice(states, p=InStDi)

            ####### create an episode of horizon H ######
            for t in range(H):
                beh_action,beh_NeSt,beh_rew = self.data_sample(states, P, reward, beh_pol[t], beh_InSt)
                # tar_InSt,tar_action,tar_NeSt,tar_rew = self.data_sample(states, P, reward, tar_pol, tar_InSt)
                beh_trajectory.append((beh_InSt,beh_action,beh_NeSt,beh_rew))
                # tar_trajectory.append((tar_InSt,tar_action,tar_NeSt,tar_rew))
                beh_InSt = beh_NeSt
                # tar_InSt = tar_NeSt
            
            beh_data.append(beh_trajectory)
            # tar_data.append(tar_trajectory)

        # return beh_data, tar_data, P, reward, beh_pol, tar_pol
        # return beh_data, P, reward, beh_pol, tar_pol
        return beh_data, P, reward, beh_pol
    
        # S, A, P, R, Mu = MDP(n_s, n_a, H)

    # def policy(actions, m):

    #     a = list(itertools.combinations(actions, m))
    #     a_dist = np.ones(len(a))/len(a)
    #     rnd = np.random.choice(np.arange(len(a)), p=a_dist)
    #     a_m = a[rnd]
    #     # print(a_m)
    #     a_m_dist = np.ones(len(a_m))/len(a_m)
    #     a = np.random.choice(a_m,1,p=a_m_dist)

    #     return a

    # def policy(actions, m):
        
    #     a_m = random.sample(list(actions), m)
    #     a = random.sample(a_m, 1)
        
    #     return a

######### from ORL_Policy_Evaluation import STKE ###########
# def STKE(n_s, n_a, data, t, K):
#     n_t = np.zeros((n_s,n_a))
#     emp_P = np.zeros((n_s,n_a,n_s))
#     r = np.zeros((n_s,n_a))
#     for k in range(K):
#         s_b, a_b = data[k][t][0], data[k][t][1][0]
#         n_t[s_b,a_b] += 1
#     for k in range(K):
#         s_b, a_b = data[k][t][0], data[k][t][1][0]
#         r[s_b,a_b] = data[k][t][3]
#         s1_b = data[k][t][2]
#         # n_t[s_b,a_b] += 1
#         emp_P[s_b,a_b,s1_b] += 1/n_t[s_b, a_b]
#     return n_t, n_t/K, emp_P, r

# def main():

#     n_s = 50
#     n_a = 50
#     m   = 10
#     H   = 5
#     K   = 1000

#     beh_data, beh_P, beh_R, beh_pol = OfflineData.data_gen(K, m, n_s, n_a, H)
#     for t in range(H):
#         n_t, rho_t, P_est, r_t = STKE(n_s, n_a, beh_data, t, K)
#         print(np.sum(n_t>0))
        
        
#         beh_SAOM = (1/(m*n_s))*np.random.binomial(1, m/n_s, (n_s,n_a))

# if __name__ == '__main__':
#     main()

