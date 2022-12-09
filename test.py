import arms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tracker import Tracker2, SWTracker, DiscountTracker
from MAB import GenericMAB as GMAB
 
from generate_data import *
from utils import plot_mean_arms, traj_arms
marker_list = ["o","v","*","s"]
color_list = ['blue','red','orange', 'c', 'm', 'green']


# Fixed standard deviation $\sigma = 0.5$

T=10000
K=5
m=2000
N=100
seed=0
##keep the distribution of arms consistent each run
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



nb_change = int(T/m)
Gamma_T_garivier = 10
reward_u_p = 1
sigma_max = 0.5
gamma_EXP3 = min(1, np.sqrt(mab.nb_arms*(nb_change*np.log(mab.nb_arms*T)+np.exp(1))/((np.exp(1)-1)*T)))
gamma_D_UCB = 1 - 1/(4*reward_u_p)*np.sqrt(Gamma_T_garivier/T)
gamma_D_UCB_unb = 1 - 1/(4*(reward_u_p+ 2*sigma_max))*np.sqrt(Gamma_T_garivier/T)
tau_theorique = 2*reward_u_p*np.sqrt(T*np.log(T)/Gamma_T_garivier)
tau_theorique_unb = 2*(reward_u_p + 2*sigma_max)*np.sqrt(T*np.log(T)/Gamma_T_garivier)
tau_no_log = 2*reward_u_p*np.sqrt(T/Gamma_T_garivier)


print('gamma_D_UCB:', 1/(1-gamma_D_UCB))
print('tau:', tau_theorique)

reg_UCB1_n_2 = mab.MC_regret('UCB1', N, T, {'C': sigma_max* np.sqrt(2)},store_step=1)
reg_EXP3S_n_2 = mab.MC_regret("EXP3S", N, T, {'alpha':1/T, 'gamma': gamma_EXP3}, store_step=1)
reg_DUCB_n_2 = mab.MC_regret('D_UCB', N, T, {'B':sigma_max*1/2,'ksi':2, 'gamma': gamma_D_UCB_unb}, store_step=1)
reg_SWUCB_n_2 = mab.MC_regret('SW_UCB', N, T, {'C': sigma_max*np.sqrt(2), 'tau': int(tau_theorique_unb)},store_step=1)
reg_SW_TS_n_2 = mab.MC_regret('SW_TS_gaussian', N, T, {'mu_0':0.5, 'sigma_0':0.5, 'sigma':0.5, 'tau': int(tau_theorique)},store_step=1)
#reg_D_TS_n_2 = mab.MC_regret('DTS_gaussian', N, T, {'mu_0':0.5, 'sigma_0':0.5, 'sigma':0.5, 'gamma': gamma_D_UCB},store_step=1)
reg_D_TS_n_2 = mab.MC_regret('DTS_gaussian1', N, T, {'kexi':100, 'tao_max':0.2, 'gamma': 0.95},store_step=1)
reg_LBSDA_n_2 = mab.MC_regret('LB_SDA', N, T, {'tau': int(tau_theorique)}, store_step=1)




#plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc("lines", linewidth=3)
# matplotlib.rc('xtick', labelsize=15)
# matplotlib.rc('ytick', labelsize=15)
# matplotlib.rc('font', weight='bold')
#matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath} \boldmath"]


t_saved = [i for i in range(T)]


#for keys in chg_dist:
#    print(keys)
#    plt.axvline(keys, color='red', linestyle='--', lw=1)
    
    
#plt.figure(figsize=(4,3))


alpha1=0.5
plt.plot(reg_UCB1_n_2[0], color = "orange", marker = "*", markevery = 1000, markersize = 8, label = 'UCB1')
plt.fill_between(t_saved, np.quantile(reg_UCB1_n_2[1], q= 0.25, axis=1), np.quantile(reg_UCB1_n_2[1], q= 0.75, axis=1), 
                 alpha=alpha1, linewidth=1.5, color="orange")

plt.plot(reg_LBSDA_n_2[0], color = "blue",  linewidth=2.5, label= "SW-LB-SDA")
plt.fill_between(t_saved, np.quantile(reg_LBSDA_n_2[1], q= 0.25, axis=1), 
                 np.quantile(reg_LBSDA_n_2[1], q= 0.75, axis=1), color = "blue",  linewidth=1.5, alpha=alpha1)
plt.plot(reg_EXP3S_n_2[0], color = "green",marker = 'o', markevery= 1000, label = 'EXP3S')
plt.fill_between(t_saved, np.quantile(reg_EXP3S_n_2[1], q= 0.25, axis=1), 
                 np.quantile(reg_EXP3S_n_2[1], q= 0.75, axis=1), color = "green",  linewidth=1.5, alpha=alpha1)
plt.plot(reg_DUCB_n_2[0], color = "m", marker= "v", linewidth = 1.5, markevery=1000, label = 'D-kl-UCB')
plt.fill_between(t_saved, np.quantile(reg_DUCB_n_2[1], q= 0.25, axis=1), 
                 np.quantile(reg_DUCB_n_2[1], q= 0.75, axis=1), color = "m",  linewidth=1.5, alpha=alpha1)

plt.plot(reg_SWUCB_n_2[0], color = "c", linestyle = "--", markevery=1000,label= "SW-kl-UCB")
plt.fill_between(t_saved, np.quantile(reg_SWUCB_n_2[1], q= 0.25, axis=1), 
                 np.quantile(reg_SWUCB_n_2[1], q= 0.75, axis=1), color = "c",  linewidth=1.5, alpha=alpha1)

plt.plot(reg_SW_TS_n_2[0], color = "red",  linewidth = 1.8, 
         linestyle = "-", marker = 's', markevery=1000,label= "SW-TS")
plt.fill_between(t_saved, np.quantile(reg_SW_TS_n_2[1], q= 0.25, axis=1), 
                 np.quantile(reg_SW_TS_n_2[1], q= 0.75, axis=1), color = "red",  linewidth=1.5, alpha=alpha1)
plt.plot(reg_D_TS_n_2[0], color = "black", linewidth = 1.5, linestyle="--",label= "D-TS")
plt.fill_between(t_saved, np.quantile(reg_D_TS_n_2[1], q= 0.25, axis=1), 
                 np.quantile(reg_D_TS_n_2[1], q= 0.75, axis=1), color = "black",  linewidth=1.5, alpha=alpha1)


    
#plt.legend(loc=2, fontsize=8).draw_frame(True)
plt.legend()
plt.xlabel('Round t')
plt.ylabel('Regret')
plt.show()
plt.savefig('final_gaussian.png')