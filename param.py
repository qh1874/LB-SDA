import numpy as np

param={
    'T':100000, # round 
    'K':200,  # arm
    'm':20000, # length of stationary phase, breakpoints=T/m
    'N':1 # repeat times
}

T=param['T']
K=param['K']
m=param['m']
nb_change=int(T/m)
Gamma_T_garivier = max(nb_change-1,1)
reward_u_p = 1
sigma_max = 1
gamma_EXP3 = min(1, 0.1*np.sqrt(K*(nb_change*np.log(K*T)+np.exp(1))/((np.exp(1)-1)*T)))
gamma_D_UCB = 1 - 1/(4*reward_u_p)*np.sqrt(Gamma_T_garivier/T)
gamma_D_UCB_unb = 1 - 1/(4*(reward_u_p+ 2*sigma_max))*np.sqrt(Gamma_T_garivier/T)
tau_theorique = 2*reward_u_p*np.sqrt(T*np.log(T)/Gamma_T_garivier)
tau_theorique_unb = 2*(reward_u_p + 2*sigma_max)*np.sqrt(T*np.log(T)/Gamma_T_garivier)
tau_no_log = 2*reward_u_p*np.sqrt(T/Gamma_T_garivier)


param_ucb1={'C': sigma_max* np.sqrt(2)}
param_exp3s={'alpha':1/T, 'gamma': gamma_EXP3}
param_dsucb={'B':sigma_max*1/2,'ksi':2, 'gamma': gamma_D_UCB_unb}
param_swucb={'C': sigma_max*np.sqrt(2), 'tau': int(tau_theorique_unb)}
param_swts={'mu_0':0.5, 'sigma_0':0.5, 'sigma':0.5, 'tau': int(tau_theorique)}
param_dsts={'kexi':100, 'tao_max':0.18, 'gamma': 0.99}
param_lbsda={'tau': int(tau_theorique)}
