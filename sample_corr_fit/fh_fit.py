from __future__ import print_function

import matplotlib.pyplot as plt
import tables as h5
import scipy as sp
import numpy as np
np.set_printoptions(linewidth=180)
import fit_functions as fit_fh
import iminuit as mn
import random
import tqdm
import theano as th
import theano.tensor as Tn


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

params = dict()
params['tau'] = 1
params['a09m310'] = dict()
params['a09m310']['seed'] = 'a09m310'
params['bs'] = False
params['a09m310']['Nbs']  = 5000
params['a09m310']['t_min_max'] = {
    'proton':[9,16],
    'gA'    :[3,12],
    'gV'    :[7,17]
}
params['a09m310']['fit_ini'] = {
    'E_0'   :.49241,
    'dE_10' :.408,
    'zs_0'  :2.2e-5,
    'zp_0'  :2.3e-3,
    'zs_1'  :1.3e-5,
    'zp_1'  :3.1e-3,
    'gA_00' :1.27,
    'gV_00' :1.02,
    'gA_11' :1.55,
    'gA_10' :-.01,
    'dAss_0':-4.7e-10,
    'dAps_0':-6.2e-8,
    'dAss_1':-4.2e-10,
    'dAps_1':1.5e-8,
    'gV_11' :1.06,
    'gV_10' :-0.066,
    'dVss_0':3.9e-10,
    'dVps_0':3.1e-8,
    'dVss_1':2.6e-11,
    'dVps_1':-4.1e-10,
}

def get_data(ens,params,verbose=False):
    t_p_i ,t_p_f  = params[ens]['t_min_max']['proton']
    t_gA_i,t_gA_f = params[ens]['t_min_max']['gA']
    t_gV_i,t_gV_f = params[ens]['t_min_max']['gV']
    tau = params['tau']
    dfile = h5.open_file('callat_gA.a09m310.h5')
    proton = dfile.get_node('/proton/'+ens).read()
    gA     = dfile.get_node('/gA/'+ens).read()
    gV     = dfile.get_node('/gV/'+ens).read()
    dfile.close()
    Ncfg,Nt,Nsnk = proton.shape
    Nbs = params[ens]['Nbs']
    Mbs = proton.shape[0]
    if verbose:
        print('collecting %s data' %ens)
        print('found Ncfg=%d, Nt=%d, Nsink=%d' %(Ncfg,Nt,Nsnk))
    total_nt = 2*(t_p_f+1-t_p_i +t_gA_f+1-t_gA_i +t_gV_f+1-t_gV_i)
    random.seed(params[ens]['seed'])
    seed = random.randint(0,1e5)
    p_bs = fit_fh.bs_corr(proton,Nbs=Nbs,Mbs=Mbs,seed=seed)
    gA_bs     = fit_fh.bs_corr(gA,Nbs=Nbs,Mbs=Mbs,seed=seed)
    gV_bs     = fit_fh.bs_corr(gV,Nbs=Nbs,Mbs=Mbs,seed=seed)
    y = np.zeros([total_nt])
    y_bs = np.zeros([Nbs,total_nt])
    nt_2p = t_p_f+1  - t_p_i
    nt_gA = t_gA_f+1 - t_gA_i
    nt_gV = t_gV_f+1 - t_gV_i

    fh_gA  = np.roll(gA.mean(axis=0)/proton.mean(axis=0),-tau,axis=0)/tau
    fh_gA += -gA.mean(axis=0)/proton.mean(axis=0)/tau
    fh_gV  = np.roll(gV.mean(axis=0)/proton.mean(axis=0),-tau,axis=0)/tau
    fh_gV += -gV.mean(axis=0)/proton.mean(axis=0)/tau
    y[                    0 : nt_2p                ] = proton.mean(axis=0)[t_p_i:t_p_f+1,0]
    y[                nt_2p : 2*nt_2p              ] = proton.mean(axis=0)[t_p_i:t_p_f+1,1]
    y[              2*nt_2p : 2*nt_2p+nt_gA        ] = fh_gA[t_gA_i:t_gA_f+1,0]
    y[        2*nt_2p+nt_gA : 2*(nt_2p+nt_gA)      ] = fh_gA[t_gA_i:t_gA_f+1,1]
    y[      2*(nt_2p+nt_gA) : 2*(nt_2p+nt_gA)+nt_gV] = fh_gV[t_gV_i:t_gV_f+1,0]
    y[2*(nt_2p+nt_gA)+nt_gV : 2*(nt_2p+nt_gA+nt_gV)] = fh_gV[t_gV_i:t_gV_f+1,1]

    fh_gA_bs  = np.roll(gA_bs/p_bs,-tau,axis=1)/tau -gA_bs/p_bs/tau
    fh_gV_bs  = np.roll(gV_bs/p_bs,-tau,axis=1)/tau -gV_bs/p_bs/tau
    y_bs[:,                    0 : nt_2p                ] = p_bs[:,t_p_i:t_p_f+1,0]
    y_bs[:,                nt_2p : 2*nt_2p              ] = p_bs[:,t_p_i:t_p_f+1,1]
    y_bs[:,              2*nt_2p : 2*nt_2p+nt_gA        ] = fh_gA_bs[:,t_gA_i:t_gA_f+1,0]
    y_bs[:,        2*nt_2p+nt_gA : 2*(nt_2p+nt_gA)      ] = fh_gA_bs[:,t_gA_i:t_gA_f+1,1]
    y_bs[:,      2*(nt_2p+nt_gA) : 2*(nt_2p+nt_gA)+nt_gV] = fh_gV_bs[:,t_gV_i:t_gV_f+1,0]
    y_bs[:,2*(nt_2p+nt_gA)+nt_gV : 2*(nt_2p+nt_gA+nt_gV)] = fh_gV_bs[:,t_gV_i:t_gV_f+1,1]

    return y,y_bs

class Chisq():
    def __init__(self,y,cov_inv,ens,params):
        self.y = y
        self.cov_inv = cov_inv
        self.t_p_i ,self.t_p_f  = params[ens]['t_min_max']['proton']
        self.t_gA_i,self.t_gA_f = params[ens]['t_min_max']['gA']
        self.t_gV_i,self.t_gV_f = params[ens]['t_min_max']['gV']
        self.nt0 = self.t_p_f+1  - self.t_p_i
        self.nt1 = self.t_gA_f+1 - self.t_gA_i
        self.nt2 = self.t_gV_f+1 - self.t_gV_i
        self.t0  = np.arange(self.t_p_i,self.t_p_f+1)
        self.t1  = np.arange(self.t_gA_i,self.t_gA_f+1)
        self.t2  = np.arange(self.t_gV_i,self.t_gV_f+1)
        self.tau   = params['tau']

    def __call__(self,\
        E_0,dE_10,\
        zs_0,zp_0,zs_1,zp_1,\
        gA_00,gA_11,gA_10,\
        gV_00,gV_11,gV_10,\
        dAss_0,dAps_0,dAss_1,dAps_1,\
        dVss_0,dVps_0,dVss_1,dVps_1):
        f = np.zeros_like(self.y)
        # 2pt SS
        f[                              0 : self.nt0                       ] = \
            fit_fh.c2pt(self.t0,E_0,zs_0,zs_0,\
                dE_10=dE_10,snk_1=zs_1,src_1=zs_1)
        # 2pt PS
        f[                       self.nt0 : 2*self.nt0                     ] = \
            fit_fh.c2pt(self.t0,E_0,zp_0,zs_0,\
                dE_10=dE_10,snk_1=zp_1,src_1=zs_1)
        # gA SS
        f[                     2*self.nt0 : 2*self.nt0 +self.nt1           ] = \
            fit_fh.fh_derivative(self.t1,self.tau,E_0,zs_0,zs_0,gA_00,dAss_0,\
                dE_10=dE_10, snk_1=zs_1, src_1=zs_1, g_11=gA_11, g_10=gA_10,d_1=dAss_1)
        # gA PS
        f[           2*self.nt0 +self.nt1 : 2*(self.nt0+self.nt1)          ] = \
            fit_fh.fh_derivative(self.t1,self.tau,E_0,zp_0,zs_0,gA_00,dAps_0,\
                dE_10=dE_10, snk_1=zp_1, src_1=zs_1, g_11=gA_11, g_10=gA_10,d_1=dAps_1)
        # gV SS
        f[          2*(self.nt0+self.nt1) : 2*(self.nt0+self.nt1) +self.nt2] = \
            fit_fh.fh_derivative(self.t2,self.tau,E_0,zs_0,zs_0,gV_00,dVss_0,\
                dE_10=dE_10, snk_1=zs_1, src_1=zs_1, g_11=gV_11, g_10=gV_10,d_1=dVss_1)
        # gV PS
        f[2*(self.nt0+self.nt1) +self.nt2 : 2*(self.nt0+self.nt1+self.nt2)] = \
            fit_fh.fh_derivative(self.t2,self.tau,E_0,zp_0,zs_0,gV_00,dVps_0,\
                dE_10=dE_10, snk_1=zp_1, src_1=zs_1, g_11=gV_11, g_10=gV_10,d_1=dVps_1)

        dy = f - self.y
        return np.dot(dy,np.dot(self.cov_inv,dy))

def fit(ens,params):
    y,y_bs  = get_data(ens,params)
    cov     = np.cov(y_bs,rowvar=False)
    cov_inv = sp.linalg.inv(cov)

    chisq_fh = Chisq(y,cov_inv,ens,params)

    ini_vals = dict()
    for k in params[ens]['fit_ini']:
        ini_vals[k] = params[ens]['fit_ini'][k]
        if 'dA' in k or 'dV' in k:
            ini_vals['error_'+k] = 0.05*params[ens]['fit_ini'][k]
        else:
            ini_vals['error_'+k] = 0.05*params[ens]['fit_ini'][k]
    ini_vals['limit_dE_10'] = (0,10)
    ini_vals['limit_zs_0'] = (0,1)
    ini_vals['limit_zs_1'] = (0,1)
    ini_vals['limit_gA_11'] = (-10,10)
    ini_vals['limit_gA_10'] = (-5,5)
    ini_vals['limit_gV_11'] = (-10,10)
    ini_vals['limit_gV_10'] = (-5,5)

    min_fh = mn.Minuit(chisq_fh, pedantic=False, print_level=1,**ini_vals)
    min_fh.migrad()
    #min_fh.minos()

    lam = []
    for k in ini_vals:
        if 'error' not in k and 'limit' not in k:
            lam.append(k)
    dof = len(y) - len(lam)
    print("chi^2 = %.4f, dof = %d, Q=%.4f" %(min_fh.fval,dof,fit_fh.p_val(min_fh.fval,dof)))

    if params['bs']:
        ini_vals_bs = dict(ini_vals)
        bs_lams = dict()
        for k in min_fh.values:
            ini_vals_bs[k] = min_fh.values[k]
            ini_vals['error_'+k] = 0.2*min_fh.errors[k]
            bs_lams[k] = np.zeros([params['a09m310']['Nbs']])
        bs_fits = []
        #for bs in tqdm.tqdm(range(params[ens]['Nbs']),desc='Nbs'):
        for bs in tqdm.tqdm(range(params['a09m310']['Nbs']),desc='Nbs'):
            chisq_fh = Chisq(y_bs[bs],cov_inv,ens,params)
            min_fh_bs = mn.Minuit(chisq_fh, pedantic=False, print_level=0,**ini_vals_bs)
            min_fh_bs.migrad()
            bs_fits.append(min_fh_bs)
            for k in min_fh_bs.values:
                bs_lams[k][bs] = min_fh_bs.values[k]
        print(bs_lams['gA_00'].mean(),bs_lams['gA_00'].std())
    return min_fh

def plot_results(ens,params,mn):
    lam_all = mn.parameters
    p_all = {k:mn.values[k] for k in lam_all}
    l_ss = ['E_0','dE_10','zs_0','zs_1']
    l_ps = ['E_0','dE_10','zs_0','zp_0','zs_1','zp_1']
    l_ga_ss = ['E_0','dE_10','zs_0','zs_1','gA_00','gA_11','gA_10','dAss_0','dAss_1']
    l_ga_ps = ['E_0','dE_10','zs_0','zp_0','zs_1','zp_1',\
        'gA_00','gA_11','gA_10','dAss_0','dAps_0','dAss_1','dAps_1']
    l_gv_ss = ['E_0','dE_10','zs_0','zs_1','gV_00','gV_11','gV_10','dVss_0','dVss_1']
    l_gv_ps = ['E_0','dE_10','zs_0','zp_0','zs_1','zp_1',\
        'gV_00','gV_11','gV_10','dVss_0','dVps_0','dVss_1','dVps_1']

    i_ss = [i for i,l in enumerate(lam_all) if l not in l_ss]
    i_ps = [i for i,l in enumerate(lam_all) if l not in l_ps]
    i_ga_ss = [i for i,l in enumerate(lam_all) if l not in l_ga_ss]
    i_ga_ps = [i for i,l in enumerate(lam_all) if l not in l_ga_ps]
    i_gv_ss = [i for i,l in enumerate(lam_all) if l not in l_gv_ss]
    i_gv_ps = [i for i,l in enumerate(lam_all) if l not in l_gv_ps]

    cov_param = np.array(mn.matrix(correlation=False))
    cov_p_ss = np.delete(np.delete(cov_param,i_ss,axis=0),i_ss,axis=1)
    cov_p_ps = np.delete(np.delete(cov_param,i_ps,axis=0),i_ps,axis=1)
    cov_ga_ss = np.delete(np.delete(cov_param,i_ga_ss,axis=0),i_ga_ss,axis=1)
    cov_ga_ps = np.delete(np.delete(cov_param,i_ga_ps,axis=0),i_ga_ps,axis=1)
    cov_gv_ss = np.delete(np.delete(cov_param,i_gv_ss,axis=0),i_gv_ss,axis=1)
    cov_gv_ps = np.delete(np.delete(cov_param,i_gv_ps,axis=0),i_gv_ps,axis=1)

    y,y_bs  = get_data(ens,params)
    t_p_i ,t_p_f  = params[ens]['t_min_max']['proton']
    t_p = t_p_f+1-t_p_i
    p_ss = y[0:t_p]
    p_ss_bs = y_bs[:,0:t_p]
    p_ps = y[t_p:2*t_p]
    p_ps_bs = y_bs[:,t_p:2*t_p]
    eff_ps = np.log(p_ps / np.roll(p_ps,-1))
    eff_ps_bs = np.log(p_ps_bs / np.roll(p_ps_bs,-1,axis=1))
    eff_ss = np.log(p_ss / np.roll(p_ss,-1))
    eff_ss_bs = np.log(p_ss_bs / np.roll(p_ss_bs,-1,axis=1))

    t_2p = np.arange(t_p_i,t_p_f+.1,.1)
    dt = t_2p[1] - t_2p[0]

    t,tau,e0,zs0,zp0,de10,zs1,zp1 = Tn.dscalars('t','tau','e0','zs0','zp0','de10','zs1','zp1')
    gA00,gA11,gA10,gV00,gV11,gV10 = Tn.dscalars('gA00','gA11','gA10','gV00','gV11','gV10')
    ds0,dp0,ds1,dp1               = Tn.dscalars('ds0','dp0','ds1','dp1')

    th_2pt_ss = zs0**2 *Tn.exp(-t*e0) + zs1**2 *Tn.exp(-t*(e0+de10))
    th_2pt_tau_ss = zs0**2 *Tn.exp(-(t+tau)*e0) + zs1**2 *Tn.exp(-(t+tau)*(e0+de10))
    th_eff_ss = Tn.log( th_2pt_ss / th_2pt_tau_ss ) / tau
    th_deff_ss = Tn.grad(th_eff_ss,[e0,de10,zs0,zs1])
    th_deff_ss_def = th.function([t,tau,e0,de10,zs0,zs1],th_deff_ss)
    th_deff_ss_fun = lambda t,tau: \
        th_deff_ss_def(t,tau,p_all['E_0'],p_all['dE_10'],p_all['zs_0'],p_all['zs_1'])
    '''
    th_2pt_ps = zp0*zs0*Tn.exp(-t*e0) + zp1*zs1*Tn.exp(-t*(e0+de10))
    th_2pt_tau_ps = zp0*zs0*Tn.exp(-(t+tau)*e0) + zp1*zs1*Tn.exp(-(t+tau)*(e0+de10))
    th_eff_ps = Tn.log( th_2pt_ps / th_2pt_tau_ps ) / tau
    th_deff_ps = Tn.grad(th_eff_ps,[e0,zp0,zs0,de10,zp1,zs1])
    th_deff_ps_def = th.function([t,tau,e0,zp0,zs0,de10,zp1,zs1],th_deff_ps)
    th_deff_ps_fun = lambda t,tau: \
        th_deff_ps_def(t,tau,p_all['E_0'],p_all['zp_0'],p_all['zs_0'],\
            p_all['dE_10'],p_all['zp_1'],p_all['zs_1'])
    p_ps = dict(p_all)
    p_ps['snk_0'] = p_ps['zp_0']
    p_ps['snk_1'] = p_ps['zp_1']
    p_ps['src_0'] = p_ps['zs_0']
    p_ps['src_1'] = p_ps['zs_1']
    ps_fit = fit_fh.c2pt(t_2p,**p_ps)
    err_ps = np.zeros_like(t_2p)
    eff_fit_ps = np.log(ps_fit / np.roll(ps_fit,-1)) / dt
    '''
    parr = dict()
    for p in l_ss:
        if 'zs' in p:
            n = p.split('_')[-1]
            parr['src_'+n] = p_all[p]
            parr['snk_'+n] = p_all[p]
        else:
            parr[p] = p_all[p]
    print(parr)
    ss_fit = fit_fh.c2pt(t_2p,**parr)
    eff_fit_ss = np.log(ss_fit / np.roll(ss_fit,-1)) / dt
    eff_fit_ss[-1] = eff_fit_ss[-2] 
    err_ss = np.zeros_like(t_2p)
    for i,t in enumerate(t_2p):
        #err_ps[i] = np.sqrt(np.dot(th_deff_ps_fun(t,dt),np.dot(cov_p_ps,th_deff_ps_fun(t,dt))))
        err_ss[i] = np.sqrt(np.dot(th_deff_ss_fun(t,dt),np.dot(cov_p_ss,th_deff_ss_fun(t,dt))))

    fig = plt.figure('2pt')
    ax = plt.axes([.14,.14,.8,.8])
    #ax.fill_between(t_2p,eff_fit_ps-err_ps,eff_fit_ps+err_ps,color='r',alpha=.3)
    ax.fill_between(t_2p,eff_fit_ss-err_ss,eff_fit_ss+err_ss,color='k',alpha=.3)
    ax.errorbar(np.arange(t_p_i,t_p_f+1),eff_ps,yerr=eff_ps_bs.std(axis=0),linestyle='None',\
        color='r',marker='s',mfc='None',mec='r')
    ax.errorbar(np.arange(t_p_i,t_p_f+1),eff_ss,yerr=eff_ss_bs.std(axis=0),linestyle='None',\
        color='k',marker='s',mfc='None',mec='k')
    ax.axis([0,20,0.4,.6])

if __name__ == "__main__":
    plt.ion()

    min_fh = fit('a09m310',params)
    plot_results('a09m310',params,min_fh)

    plt.ioff()
    if run_from_ipython():
        plt.show(block=False)
    else:
        plt.show()
