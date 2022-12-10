import arms
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from tracker import Tracker2, SWTracker, DiscountTracker
from MAB import GenericMAB as GMAB
 
from generate_data import get_reward_distribution
from utils import plot_mean_arms, traj_arms,save_data
import time
from param import *

# Fixed standard deviation $\sigma = 0.5$

T=param['T'] # Number of rounds
K=param['K'] # Number of Arms
m=param['m'] # Length of stationary phase, breakpoints=T/m
N=param['N'] # Repeat Times


# Keep the distribution of arms consistent each run
seed=1
#seed=np.random.randint(0,1000)
test_normal,r_opt=get_reward_distribution(T,K,m,seed)
KG=['G' for _ in range(K)]
arm_start=KG
param_start=test_normal[0].tolist()
chg_dist={}
for i in range(m,T,m):
    chg_dist[str(i)]=[ KG,test_normal[i].tolist() ]

# arm_start, param_start =['G', 'G', 'G'], [[0.9,0.5], [0.5,0.5], [0.4,0.5]]
# chg_dist = {'2500': [['G', 'G', 'G'], [[0.4,0.5], [0.8,0.5], [0.5,0.5]]],
#             '4500': [['G', 'G', 'G'], [[0.3,0.5], [0.2,0.5], [0.7,0.5]]],
#             '7000': [['G', 'G', 'G'], [[0.9,0.5], [0.8,0.5], [0.4,0.5] ]]
#            }

mab = GMAB(arm_start, param_start, chg_dist)

UCB1_data = mab.MC_regret('UCB1', N, T, param_ucb1,store_step=1)
EXP3S_data = mab.MC_regret("EXP3S", N, T, param_exp3s, store_step=1)
DS_UCB_data = mab.MC_regret('DS_UCB', N, T, param_dsucb, store_step=1)
SW_UCB_data = mab.MC_regret('SW_UCB', N, T, param_swucb,store_step=1)
SW_TS_data = mab.MC_regret('SW_TS_gaussian', N, T, param_swts,store_step=1)
#DS_TS_data = mab.MC_regret('DTS_gaussian', N, T, {'mu_0':0.5, 'sigma_0':0.5, 'sigma':0.5, 'gamma': gamma_D_UCB},store_step=1)
DS_TS_data = mab.MC_regret('DS_TS_gaussian', N, T, param_dsts,store_step=1)
TS_data = mab.MC_regret('TS_gaussian', N, T,{},store_step=1)
LBSDA_data = mab.MC_regret('LB_SDA', N, T, param_lbsda, store_step=1)

L=['UCB1_data','EXP3S_data','DS_UCB_data','SW_UCB_data','SW_TS_data','DS_TS_data','TS_data','LBSDA_data']

x=np.arange(T)
d = int(T / 20)
xx = np.arange(0, T, d)
alpha=0.05

low_bound, high_bound = sms.DescrStatsW(EXP3S_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, EXP3S_data[0][xx], '-g^', markerfacecolor='none', label='EXP3S')
plt.fill_between(x, low_bound, high_bound, alpha=0.5,color='g')

low_bound, high_bound = sms.DescrStatsW(LBSDA_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, LBSDA_data[0][xx], '-c*', markerfacecolor='none', label='SW-LB-SDA')
plt.fill_between(x, low_bound, high_bound, alpha=0.5,color='c')

low_bound, high_bound = sms.DescrStatsW(UCB1_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, UCB1_data[0][xx], '-k^', markerfacecolor='none', label='UCB1')
plt.fill_between(x, low_bound, high_bound, alpha=0.5,color='k')

low_bound, high_bound = sms.DescrStatsW(DS_UCB_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, DS_UCB_data[0][xx], '-bd', markerfacecolor='none', label='DS_kl-UCB')
plt.fill_between(x, low_bound, high_bound, alpha=0.5,color='b')

low_bound, high_bound = sms.DescrStatsW(SW_UCB_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, SW_UCB_data[0][xx], '-ms', markerfacecolor='none', label='SW_kl_UCB')
plt.fill_between(x, low_bound, high_bound, alpha=0.5,color='m')

low_bound, high_bound = sms.DescrStatsW(TS_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, TS_data[0][xx], color='brown',marker='*', markerfacecolor='none', label='TS')
plt.fill_between(x, low_bound, high_bound, alpha=0.5,color='brown')

low_bound, high_bound = sms.DescrStatsW(SW_TS_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, SW_TS_data[0][xx], '-y^', markerfacecolor='none', label='SW_TS')
plt.fill_between(x, low_bound, high_bound, alpha=0.5,color='y')

low_bound, high_bound = sms.DescrStatsW(DS_TS_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, DS_TS_data[0][xx], '-ro', markerfacecolor='none', label='DS_TS')
plt.fill_between(x, low_bound, high_bound, alpha=0.5,color='r')


plt.legend()
plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
plt.xlabel('Round t')
plt.ylabel('Regret')
plt.show()
#plt.savefig('final_gaussian.png')