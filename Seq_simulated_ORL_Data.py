# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:35:23 2024

@author: Komal
"""

import numpy as np
import random
# import json
import time

class OfflineData:
    def __init__(self, n_s, n_a, m, h, K, rank, reward_level):
        self.n_s = n_s
        self.n_a = n_a
        self.m = m
        self.H = h
        self.K = K
        self.r = rank
        self.reward_level = reward_level
        self.beh_data = []

    def MDP(self):

        states = np.arange(self.n_s)
        actions = np.arange(self.n_a)
        
        InStDi = np.ones(self.n_s)/self.n_s

        np.random.seed(26)

        # P = np.ones((self.n_s,self.n_a,self.n_s))/self.n_s

        reward = []
        beh_policy = []
        P = np.ones((self.n_s,self.n_a,self.n_s))/self.n_s
        for _ in range(self.H):
            ######## generating rank-1 reward and transition tensor ########
            r_t = np.random.randn(self.n_s,self.r)@np.ones((self.n_a,self.r)).T
            # P_t = np.ones((self.n_s,self.n_a,self.n_s))/self.n_s
            # P_t = np.random.rand(self.n_s, self.n_a, self.n_s)
            # for i in states:
            #     for j in actions:
            #         P_t[i, j, :] = P_t[i, j, :]/sum(P_t[i, j, :])
            beh_policy.append((1/self.m)*np.random.binomial(1, self.m/self.n_a, (self.n_s,self.n_a)))
            # P.append(P_t)
            reward.append(r_t * self.reward_level)
        
        # print(reward)
        # fjod
        
        return states, actions, P, reward, InStDi, beh_policy

    def data_sample(states, P, reward, pol, InSt):
        
        actions_available = np.nonzero(pol[InSt,:])[0]
        action = random.sample(list(actions_available), 1)
        NeSt   = np.random.choice(states, p=P[InSt,action[0]])
        rew    = reward[InSt,action[0]]

        return action, NeSt, rew

    def data_gen(self):
        
        ###### generate a MDP ######
        states, _, P, reward, InStDi, beh_pol = self.MDP()
        
        ###### sample K trajectories from generated MDP ####### 
        for _ in range(self.K):
            beh_trajectory = []
            beh_InSt = np.random.choice(states, p=InStDi)

            ####### create an episode of horizon H ######
            for t in range(self.H):
                beh_action, beh_NeSt, beh_rew = OfflineData.data_sample(states, P, reward[t], beh_pol[t], beh_InSt)
                beh_trajectory.append([beh_InSt,beh_action,beh_NeSt,beh_rew])
                beh_InSt = beh_NeSt
            
            self.beh_data.append(beh_trajectory)

        return self.beh_data #, P, reward #, beh_pol

######### from ORL_Policy_Evaluation import STKE ###########
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
        emp_P[s_b,a_b,s1_b] += 1/n_t[s_b, a_b]
    return n_t, n_t/K, emp_P, r

def main():

    n_s = 10
    n_a = 5
    M   = [5] #[50, 70, 90]
    H   = [5] #[5, 7, 9, 11, 13]
    K   = [1000*1] #[1000, 10000, 100000, 1000000, 10000000]
    rank = 1
    reward_level = 1.0
    no_trials = 1

    start_time = time.perf_counter()

    for trial in range(no_trials):
      for h in H:
        for m in M:
          for k in K:

              Data_Generation = OfflineData(n_s, n_a, m, h, k, rank, reward_level)

              beh_data = Data_Generation.data_gen()
            # print(len(beh_data))
            # print(len(beh_data[0]))
            # beh_data_array = np.array(beh_data)
            # file_path = "./behavior data for %d states %d action subset size %d trajectories horizon %d and trial %d"%(n_s, m, k, h, trial+1)
            # np.savez("./behavior data for %d states %d action subset size %d trajectories and horizon %d"%(n_s, m, k, h), beh_data)
            # json.dump(beh_data, file_path)
            # np.savetxt(file_path, X = beh_data)
    # print(beh_R)
    end_time = time.perf_counter()
    print(f"Time taken to complete {round(end_time-start_time, 3)} second(s)")
    # for t in range(H[0]):
    #     n_t, rho_t, P_est, r_t = STKE(n_s, n_a, beh_data, t, K[0])
        # print(P_est)
        # print(r_t)
        # print(n_t)

if __name__ == '__main__':
    main()
