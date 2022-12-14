import arms
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from tracker import Tracker2, SWTracker, DiscountTracker
from MAB import GenericMAB as GMAB
from generate_data import generate_arm_Gauss,generate_arm_Bernoulli
from utils import plot_mean_arms, traj_arms,save_data
from param import *

# Fixed standard deviation $\sigma = 0.5$

T=param['T'] # Number of rounds
K=param['K'] # Number of Arms
m=param['m'] # Length of stationary phase, breakpoints=T/m
N=param['N'] # Repeat Times
 

# Keep the distribution of arms consistent each run
seed=0
arm_start,param_start,chg_dist=generate_arm_Bernoulli(T,K,m,seed)

# arm_start, param_start =['G', 'G', 'G'], [[0.9,0.5], [0.5,0.5], [0.4,0.5]]
# chg_dist = {'2500': [['G', 'G', 'G'], [[0.4,0.5], [0.8,0.5], [0.5,0.5]]],
#             '4500': [['G', 'G', 'G'], [[0.3,0.5], [0.2,0.5], [0.7,0.5]]],
#             '7000': [['G', 'G', 'G'], [[0.9,0.5], [0.8,0.5], [0.4,0.5] ]]
#            }

mab = GMAB(arm_start, param_start, chg_dist)

EXP3S_data = mab.MC_regret("EXP3S", N, T, param_exp3s, store_step=1)
DS_UCB_data = mab.MC_regret('DS_UCB', N, T, param_dsucb, store_step=1)
SW_UCB_data = mab.MC_regret('SW_UCB', N, T, param_swucb,store_step=1)
SW_TS_data = mab.MC_regret('SW_TS_gaussian', N, T, param_swts,store_step=1)
DS_TS_data = mab.MC_regret('DS_TS_gaussian', N, T, param_dsts,store_step=1)
TS_data = mab.MC_regret('TS_gaussian', N, T,{},store_step=1)
LBSDA_data = mab.MC_regret('LB_SDA', N, T, param_lbsda, store_step=1)
CUSUM_data = mab.MC_regret('CUSUM', N, T, {'alpha':alpha_CUSUM , 'h': h_CUSUM, 'M':M_CUSUM, 'eps':eps_CUSUM, 'ksi':1/2},store_step=1)

L=['EXP3S_data','CUSUM_data','DS_UCB_data','SW_UCB_data','SW_TS_data','DS_TS_data','TS_data','LBSDA_data']
for i in L:
    print(i+":",eval(i)[0][-1])

x=np.arange(T)
d = int(T / 20)
xx = np.arange(0, T, d)
alpha=0.05
plt.figure(2)
low_bound, high_bound = sms.DescrStatsW(EXP3S_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, EXP3S_data[0][xx], '-g^', markerfacecolor='none', label='EXP3S')
plt.fill_between(x, low_bound, high_bound, alpha=0.5,color='g')

low_bound, high_bound = sms.DescrStatsW(LBSDA_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, LBSDA_data[0][xx], '-c*', markerfacecolor='none', label='SW-LB-SDA')
plt.fill_between(x, low_bound, high_bound, alpha=0.5,color='c')

low_bound, high_bound = sms.DescrStatsW(CUSUM_data[1].T).tconfint_mean(alpha=alpha)
plt.plot(xx, CUSUM_data[0][xx], '-k^', markerfacecolor='none', label='CUSUM')
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
