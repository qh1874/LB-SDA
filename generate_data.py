import numpy as np
import matplotlib.pyplot as plt


def get_reward_distribution_Gauss(T,K,m,seed):
    np.random.seed(seed)
    test_normal=np.zeros((T,K,2))
    xtemp=np.zeros((K,2))
    for ii in range(0,T,m):
        xtemp[:,0]=np.random.uniform(0,1,K)
        xtemp[:,1]=np.random.uniform(0,1,K)
        test_normal[ii:ii+m]=xtemp#np.random.uniform(0,1,(K,2))
        
    return test_normal


def get_reward_distribution_Bernoulli(T,K,m,seed):
    np.random.seed(seed)
    test_bernoulli=np.zeros((T,K))
    for ii in range(0,T,m):
            test_bernoulli[ii:ii+m]=np.random.uniform(0,1,K)
        
    return test_bernoulli


def get_reward_distribution_Finite(T,K,m,seed):
    np.random.seed(seed)
    dim=5
    test_finite=np.zeros((T,K,2,dim))
    for ii in range(0,T,m):
        xp=np.random.uniform(0,1,(K,2,dim))
        p=xp[:,1,:]
        ptemp=p/p.sum(1).reshape(-1,1)
        xp[:,1,:]=ptemp
        test_finite[ii:ii+m]=xp
    return test_finite


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
        plt.plot(test_normal[:,i,0],label='Arm '+str(i+1))
    plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    plt.legend()
    return arm_start,param_start,chg_dist


def generate_arm_Bernoulli(T,K,m,seed):
    test_bernoulli=get_reward_distribution_Bernoulli(T,K,m,seed)
    KB=['B' for _ in range(K)]
    arm_start=KB
    param_start=test_bernoulli[0].tolist()
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KB,test_bernoulli[i].tolist()]
    
    plt.figure(1)
    for i in range(K):
        plt.plot(test_bernoulli[:,i],label='Arm '+str(i+1))
    plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    plt.legend()
    return arm_start,param_start,chg_dist


def generate_arm_Finite(T,K,m,seed):
    test_finite=get_reward_distribution_Finite(T,K,m,seed)
    KF=['F' for _ in range(K)]
    arm_start=KF
    param_start=[list(i) for i in list(test_finite[0])]
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KF,[list(ii) for ii in list(test_finite[i])] ]
        
    plt.figure(1)
    for i in range(K):
        plt.plot(np.sum(test_finite[:,i,1,:]*test_finite[:,i,0,:],1),label='Arm '+str(i+1))
    plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    plt.legend()
    return arm_start,param_start,chg_dist