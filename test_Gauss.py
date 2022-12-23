import arms
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from tracker import Tracker2, SWTracker, DiscountTracker
from MAB import GenericMAB as GMAB
from generate_data import generate_arm_Gauss,generate_arm_Bernoulli,generate_arm_Finite
from utils import plot_mean_arms, traj_arms,save_data
from param import *


T=param['T'] # Number of rounds
K=param['K'] # Number of Arms
m=param['m'] # Length of stationary phase, breakpoints=T/m
N=param['N'] # Repeat Times
 

# Keep the distribution of arms consistent each run
seed=0
arm_start,param_start,chg_dist=generate_arm_Gauss(T,K,m,seed)

# arm_start, param_start =['G', 'G', 'G'], [[0.9,0.5], [0.5,0.5], [0.4,0.5]]
# chg_dist = {'2500': [['G', 'G', 'G'], [[0.4,0.5], [0.8,0.5], [0.5,0.5]]],
#             '4500': [['G', 'G', 'G'], [[0.3,0.5], [0.2,0.5], [0.7,0.5]]],
#             '7000': [['G', 'G', 'G'], [[0.9,0.5], [0.8,0.5], [0.4,0.5] ]]
#            }

mab = GMAB(arm_start, param_start, chg_dist)

EXP3S_data = mab.MC_regret("EXP3S", N, T, param_exp3s, store_step=1)
DS_UCB_data = mab.MC_regret('DS_UCB', N, T, param_dsucb, store_step=1)
SW_TS_data = mab.MC_regret('SW_TS_gaussian', N, T, param_swts,store_step=1)
DS_TS_data = mab.MC_regret('DS_TS_gaussian', N, T, param_dsts,store_step=1)
TS_data = mab.MC_regret('TS_gaussian', N, T,{},store_step=1)
LBSDA_data = mab.MC_regret('LB_SDA', N, T, param_lbsda, store_step=1)
CUSUM_data = mab.MC_regret('CUSUM', N, T, param_cumsum,store_step=1)
M_UCB_data = mab.MC_regret('M_UCB', N, T, param_mucb,store_step=1)

L=['EXP3S_data','CUSUM_data','DS_UCB_data','SW_TS_data','TS_data','LBSDA_data','M_UCB_data','DS_TS_data']
for i in L:
    print(i+":",eval(i)[0][-1])
print("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))

x=np.arange(T)
d = int(T / 20)
dd=int(T/1000)
xx = np.arange(0, T, d)
xxx=np.arange(0,T,dd)
alpha=0.05
plt.figure(2)
EXP3S_data1=EXP3S_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(EXP3S_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, EXP3S_data[0][xx], '-g^', markerfacecolor='none', label='EXP3S')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='g')

LBSDA_data1=LBSDA_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(LBSDA_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, LBSDA_data[0][xx], '-c*', markerfacecolor='none', label='SW-LB-SDA')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='c')

CUSUM_data1=CUSUM_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(CUSUM_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, CUSUM_data[0][xx], '-k^', markerfacecolor='none', label='CUSUM')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='k')

DS_UCB_data1=DS_UCB_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(DS_UCB_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, DS_UCB_data[0][xx], '-bd', markerfacecolor='none', label='DS-UCB')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='b')

M_UCB_data1=M_UCB_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(M_UCB_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, M_UCB_data[0][xx], '-ms', markerfacecolor='none', label='M_UCB')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='m')

TS_data1=TS_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(TS_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, TS_data[0][xx], color='brown',marker='*', markerfacecolor='none', label='TS')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='brown')

SW_TS_data1=SW_TS_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(SW_TS_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, SW_TS_data[0][xx], '-y^', markerfacecolor='none', label='SW_TS')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='y')

DS_TS_data1=DS_TS_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(DS_TS_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, DS_TS_data[0][xx], '-ro', markerfacecolor='none', label='DS_TS')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='r')

plt.legend()
#plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
plt.xlabel('Round t')
plt.ylabel('Regret')
plt.show()
