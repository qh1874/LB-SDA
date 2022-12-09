import numpy as np
from numba import jit


@jit(nopython=True)
def trunc(x, a=0, b=1):
    if x < a:
        x = a
    if x > b:
        x = b

    return x


@jit(nopython=True)
def generate_data(K,seed):
    test_normal = np.zeros((K,2))
    np.random.seed(seed)
    for i in range(K):
        mean = np.random.uniform(0.1, 1)
        test_normal[i][0] = np.random.normal(mean, 0.01)
        test_normal[i][1] = np.random.uniform(0,1)
    return test_normal


@jit(nopython=True)
def get_reward_distribution(T,K,m,seed):
    
    test_normal=np.zeros((T,K,2))
    for ii in range(0,T,m):
            test_normal[ii:ii+m]=generate_data(K,ii+seed)
    r_opt=np.zeros(T)
    for ii in range(T):
        r_opt[ii]=trunc(np.max(test_normal[ii,:,0]))
        
    return test_normal,r_opt

