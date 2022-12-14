import numpy as np
import matplotlib.pyplot as plt


def get_reward_distribution_Gauss(T,K,m,seed):
    np.random.seed(seed)
    test_normal=np.zeros((T,K,2))
    for ii in range(0,T,m):
            test_normal[ii:ii+m]=np.random.uniform(0,1,(K,2))
        
    return test_normal


def get_reward_distribution_Bernoulli(T,K,m,seed):
    np.random.seed(seed)
    test_normal=np.zeros((T,K))
    for ii in range(0,T,m):
            test_normal[ii:ii+m]=np.random.uniform(0,1,K)
        
    return test_normal


def generate_arm_Gauss(T,K,m,seed):
    test_normal=get_reward_distribution_Gauss(T,K,m,seed)
    KG=['G' for _ in range(K)]
    arm_start=KG
    param_start=test_normal[0].tolist()
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KG,test_normal[i].tolist()]
        
    plt.figure(1)
    for i in range(K):
        plt.plot(test_normal[:,i,0],label='arm'+str(i))
    plt.legend()
    return arm_start,param_start,chg_dist


def generate_arm_Bernoulli(T,K,m,seed):
    test_normal=get_reward_distribution_Bernoulli(T,K,m,seed)
    KB=['B' for _ in range(K)]
    arm_start=KB
    param_start=test_normal[0].tolist()
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KB,test_normal[i].tolist()]
    
    plt.figure(1)
    for i in range(K):
        plt.plot(test_normal[:,i],label='arm'+str(i))
    plt.legend()
    return arm_start,param_start,chg_dist
