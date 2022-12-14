import numpy as np


def get_reward_distribution(T,K,m,seed):
    np.random.seed(seed)
    test_normal=np.zeros((T,K,2))
    for ii in range(0,T,m):
            test_normal[ii:ii+m]=np.random.uniform(0,1,(K,2))
        
    return test_normal

