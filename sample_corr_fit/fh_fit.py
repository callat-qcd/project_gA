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
import fit_params as fitp

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def rgb(ens):
    if 'a15' in ens:
        clr = '#ec5d57'
    elif 'a12' in ens:
        clr ='#70bf41'
    elif 'a09' in ens:
        clr = '#51a7f9'
    return clr

ens = 'a09m310'
params = fitp.params
params['tau'] = 1
params['bs'] = False
params[ens]['seed'] = ens
params[ens]['Nbs']  = 5000
params[ens]['t_min_max'] = {
    'proton':[9,16],
    'gA'    :[3,12],
    'gV'    :[7,17]
}
params[ens]['plt_range'] = {
    'proton':[7,18,0.47,0.55],
    'gA'    :[0,15,1.1,1.5],
    'gV'    :[5,20,1.015,1.035]
}
params[ens]['fit_ini'] = {
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

def get_data(ens,params,alldata=False,verbose=False):
    t_p_i ,t_p_f  = params[ens]['t_min_max']['proton']
    t_gA_i,t_gA_f = params[ens]['t_min_max']['gA']
    t_gV_i,t_gV_f = params[ens]['t_min_max']['gV']
    tau = params['tau']
    dfile = h5.open_file('callat_gA.h5')
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

    if alldata:
        d = [proton.mean(axis=0),fh_gA,fh_gV]
        d_bs = [p_bs,fh_gA_bs,fh_gV_bs]
        return d, d_bs
    else:
        return y,y_bs

class ChisqFH():
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

    chisq_fh = ChisqFH(y,cov_inv,ens,params)

    ini_vals = dict()
    for k in params[ens]['fit_ini']:
        ini_vals[k] = params[ens]['fit_ini'][k]
        if 'dA' in k or 'dV' in k:
            ini_vals['error_'+k] = 0.005*params[ens]['fit_ini'][k]
        else:
            ini_vals['error_'+k] = 0.005*params[ens]['fit_ini'][k]
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
            chisq_fh = ChisqFH(y_bs[bs],cov_inv,ens,params)
            min_fh_bs = mn.Minuit(chisq_fh, pedantic=False, print_level=0,**ini_vals_bs)
            min_fh_bs.migrad()
            bs_fits.append(min_fh_bs)
            for k in min_fh_bs.values:
                bs_lams[k][bs] = min_fh_bs.values[k]
        print(bs_lams['gA_00'].mean(),bs_lams['gA_00'].std())
    return min_fh

def plot_results(ens,params,mn,corr,lam_lst,figname,\
    figsize=(7,7/1.618034333)):
    ''' plot data '''
    dset = {'proton':0,'gA':1,'gV':2}
    di   = dset[corr]
    clrs = ['k',rgb(ens)]
    lbl     = ['SS','PS']
    ylabel = {'proton':r'$m_{eff}(t)$','gA':r'$g_A^{FH}(t)$','gV':r'$g_V^{FH}(t)$'}
    y,y_bs  = get_data(ens,params,alldata=True)
    nt      = y[di].shape[0]

    fig = plt.figure(figname)
    ax = plt.axes([.15,.15,.8,.8])
    for i in range(y[di].shape[-1]):
        if corr == 'proton':
            eff    = np.log(y[di][:,i] / np.roll(y[di][:,i],-1))
            eff_bs = np.log(y_bs[di][:,:,i] / np.roll(y_bs[di][:,:,i],-1,axis=1))
        else:
            eff    = y[di][:,i]
            eff_bs = y_bs[di][:,:,i]
        ax.errorbar(np.arange(nt),eff,yerr=eff_bs.std(axis=0),linestyle='None',\
            color=clrs[i],marker='s',mfc='None',mec=clrs[i],label=lbl[i])
    ax.axis(params[ens]['plt_range'][corr])
    ax.set_xlabel(r'$t$',fontsize=20)
    ax.set_ylabel(ylabel[corr],fontsize=20)
    ax.legend(numpoints=1,fontsize=16,loc=2)

    ''' add Theano functions to plot fits '''
    lam_all = mn.parameters
    p_all = {k:mn.values[k] for k in lam_all}
    l_ss,l_ps = lam_lst[0],lam_lst[1]
    i_ss = [i for i,l in enumerate(lam_all) if l not in l_ss]
    i_ps = [i for i,l in enumerate(lam_all) if l not in l_ps]
    cov_param = np.array(mn.matrix(correlation=False))

    cov_ss = np.delete(np.delete(cov_param,i_ss,axis=0),i_ss,axis=1)
    cov_ps = np.delete(np.delete(cov_param,i_ps,axis=0),i_ps,axis=1)

    t_p_i ,t_p_f  = params[ens]['t_min_max']['proton']
    t_gA_i,t_gA_f = params[ens]['t_min_max']['gA']
    t_gV_i,t_gV_f = params[ens]['t_min_max']['gV']
    if corr == 'proton':
        tp = np.arange(t_p_i,t_p_f+.1,.1)
    elif corr == 'gA':
        tp = np.arange(t_gA_i,t_gA_f+.1,.1)
    elif corr == 'gV':
        tp = np.arange(t_gV_i,t_gV_f+.1,.1)
    dt = tp[1] - tp[0]

    ''' define the theano scalars we need '''
    t,tau,e0,zs0,zp0,de10,zs1,zp1 = Tn.dscalars('t','tau','e0','zs0','zp0','de10','zs1','zp1')
    g00,g11,g10                   = Tn.dscalars('g00','g11','g10')
    ds0,dp0,ds1,dp1               = Tn.dscalars('ds0','dp0','ds1','dp1')

    ''' now construct the theano functions '''
    parr_ss = dict()
    for p in l_ss:
        n = p.split('_')[-1]
        if 'zs' in p:
            parr_ss['src_'+n] = p_all[p]
            parr_ss['snk_'+n] = p_all[p]
        elif 'dA' in p or 'dV' in p:
            parr_ss['d_'+n]   = p_all[p]
        elif 'gA' in p or 'gV' in p:
            parr_ss['g_'+n]   = p_all[p]
        else:
            parr_ss[p] = p_all[p]
    parr_ps = dict()
    for p in l_ps:
        n = p.split('_')[-1]
        if 'zs' in p:
            parr_ps['src_'+n] = p_all[p]
        elif 'zp' in p:
            parr_ps['snk_'+n] = p_all[p]
        elif 'dA' in p or 'dV' in p:
            parr_ps['d_'+n]   = p_all[p]
        elif 'gA' in p or 'gV' in p:
            parr_ps['g_'+n]   = p_all[p]
        else:
            parr_ps[p] = p_all[p]
    ''' 2pt function '''
    th_ss     = zs0**2 *Tn.exp(-t*e0) + zs1**2 *Tn.exp(-t*(e0+de10))
    th_tau_ss = zs0**2 *Tn.exp(-(t+tau)*e0) + zs1**2 *Tn.exp(-(t+tau)*(e0+de10))
    th_eff_ss   = Tn.log( th_ss / th_tau_ss ) / tau
    d_th_eff_ss = Tn.grad(th_eff_ss,[e0,de10,zs0,zs1])
    d_th_eff_ss_def = th.function([t,tau,e0,de10,zs0,zs1],d_th_eff_ss)
    d_th_eff_ss_fun = lambda t,tau:\
        d_th_eff_ss_def(t,tau,p_all['E_0'],p_all['dE_10'],\
            p_all['zs_0'],p_all['zs_1'])

    th_ps = zp0*zs0*Tn.exp(-t*e0) + zp1*zs1*Tn.exp(-t*(e0+de10))
    th_tau_ps = zp0*zs0*Tn.exp(-(t+tau)*e0) + zp1*zs1*Tn.exp(-(t+tau)*(e0+de10))
    th_eff_ps = Tn.log( th_ps / th_tau_ps ) / tau
    d_th_eff_ps = Tn.grad(th_eff_ps,[e0,de10,zs0,zp0,zs1,zp1])
    d_th_eff_ps_def = th.function([t,tau,e0,de10,zs0,zp0,zs1,zp1],d_th_eff_ps)
    d_th_eff_ps_fun = lambda t,tau:\
        d_th_eff_ps_def(t,tau,p_all['E_0'],p_all['dE_10'],\
            p_all['zs_0'],p_all['zp_0'],p_all['zs_1'],p_all['zp_1'])
    ''' FH gA function '''
    g_00 = corr+'_00'
    g_11 = corr+'_11'
    g_10 = corr+'_10'
    d_ss_0 = corr.replace('g','d')+'ss_0'
    d_ss_1 = corr.replace('g','d')+'ss_1'
    d_ps_0 = corr.replace('g','d')+'ps_0'
    d_ps_1 = corr.replace('g','d')+'ps_1'

    th_num_ss  = (t-1)*zs0**2 *Tn.exp(-t*e0)*g00
    th_num_ss += (t-1)*zs1**2 *Tn.exp(-t*(e0+de10))*g11
    th_num_ss += ds0*Tn.exp(-t*e0)
    th_num_ss += ds1*Tn.exp(-t*(e0+de10))
    th_num_ss += zs0*zs1*g10*(Tn.exp(-t*e0)*Tn.exp(-de10/2) - Tn.exp(-t*(e0+de10))*Tn.exp(de10/2))\
        / (Tn.exp(de10 / 2) - Tn.exp(-de10 / 2))
    th_num_ss += zs1*zs0*g10*(Tn.exp(-t*(e0+de10))*Tn.exp(de10/2) - Tn.exp(-t*e0)*Tn.exp(-de10/2))\
        /(Tn.exp(-de10 / 2) - Tn.exp(de10 / 2))

    th_num_dt_ss  = (t-1+tau)*zs0**2 *Tn.exp(-(t+tau)*e0)*g00
    th_num_dt_ss += (t-1+tau)*zs1**2 *Tn.exp(-(t+tau)*(e0+de10))*g11
    th_num_dt_ss += ds0*Tn.exp(-(t+tau)*e0)
    th_num_dt_ss += ds1*Tn.exp(-(t+tau)*(e0+de10))
    th_num_dt_ss += zs0*zs1*g10*\
        (Tn.exp(-(t+tau)*e0)*Tn.exp(-de10/2) - Tn.exp(-(t+tau)*(e0+de10))*Tn.exp(de10/2))\
        / (Tn.exp(de10 / 2) - Tn.exp(-de10 / 2))
    th_num_dt_ss += zs1*zs0*g10*\
        (Tn.exp(-(t+tau)*(e0+de10))*Tn.exp(de10/2) - Tn.exp(-(t+tau)*e0)*Tn.exp(-de10/2))\
        /(Tn.exp(-de10 / 2) - Tn.exp(de10 / 2))

    th_fh_ss = (th_num_dt_ss / th_tau_ss - th_num_ss / th_ss) / tau
    th_fh_grad_ss = Tn.grad(th_fh_ss,[e0,de10,zs0,zs1,g00,g11,g10,ds0,ds1])
    th_grad_ss_def = th.function([t,tau,e0,de10,zs0,zs1,g00,g11,g10,ds0,ds1],th_fh_grad_ss)
    th_grad_ss_fun = lambda t,tau: \
        th_grad_ss_def(t,tau,p_all['E_0'],p_all['dE_10'],p_all['zs_0'],p_all['zs_1'],\
            p_all[g_00],p_all[g_11],p_all[g_10],\
            p_all[d_ss_0],p_all[d_ss_1])
    ''' FH PS '''
    th_num_ps  = (t-1)*zp0*zs0 *Tn.exp(-t*e0)*g00
    th_num_ps += (t-1)*zp1*zs1 *Tn.exp(-t*(e0+de10))*g11
    th_num_ps += dp0*Tn.exp(-t*e0)
    th_num_ps += dp1*Tn.exp(-t*(e0+de10))
    th_num_ps += zp0*zs1*g10*(Tn.exp(-t*e0)*Tn.exp(-de10/2) - Tn.exp(-t*(e0+de10))*Tn.exp(de10/2))\
        / (Tn.exp(de10 / 2) - Tn.exp(-de10 / 2))
    th_num_ps += zp1*zs0*g10*(Tn.exp(-t*(e0+de10))*Tn.exp(de10/2) - Tn.exp(-t*e0)*Tn.exp(-de10/2))\
        /(Tn.exp(-de10 / 2) - Tn.exp(de10 / 2))

    th_num_dt_ps  = (t-1+tau)*zp0*zs0 *Tn.exp(-(t+tau)*e0)*g00
    th_num_dt_ps += (t-1+tau)*zp1*zs1 *Tn.exp(-(t+tau)*(e0+de10))*g11
    th_num_dt_ps += dp0*Tn.exp(-(t+tau)*e0)
    th_num_dt_ps += dp1*Tn.exp(-(t+tau)*(e0+de10))
    th_num_dt_ps += zp0*zs1*g10*\
        (Tn.exp(-(t+tau)*e0)*Tn.exp(-de10/2) - Tn.exp(-(t+tau)*(e0+de10))*Tn.exp(de10/2))\
        / (Tn.exp(de10 / 2) - Tn.exp(-de10 / 2))
    th_num_dt_ps += zp1*zs0*g10*\
        (Tn.exp(-(t+tau)*(e0+de10))*Tn.exp(de10/2) - Tn.exp(-(t+tau)*e0)*Tn.exp(-de10/2))\
        /(Tn.exp(-de10 / 2) - Tn.exp(de10 / 2))

    th_fh_ps = (th_num_dt_ps / th_tau_ps - th_num_ps / th_ps) / tau
    th_fh_grad_ps = Tn.grad(th_fh_ps,[e0,de10,zs0,zp0,zs1,zp1,g00,g11,g10,dp0,dp1])
    th_grad_ps_def = th.function([t,tau,e0,de10,zs0,zp0,zs1,zp1,g00,g11,g10,dp0,dp1],th_fh_grad_ps)
    th_grad_ps_fun = lambda t,tau: \
        th_grad_ps_def(t,tau,p_all['E_0'],p_all['dE_10'],\
            p_all['zs_0'],p_all['zp_0'],p_all['zs_1'],p_all['zp_1'],\
            p_all[g_00],p_all[g_11],p_all[g_10],\
            p_all[d_ps_0],p_all[d_ps_1])

    if corr == 'proton':
        fit_ss = fit_fh.c2pt(tp,**parr_ss)
        eff_ss = np.log(fit_ss / np.roll(fit_ss,-1)) / dt
        eff_ss[-1] = eff_ss[-2]
        fit_ps = fit_fh.c2pt(tp,**parr_ps)
        eff_ps = np.log(fit_ps / np.roll(fit_ps,-1)) / dt
        eff_ps[-1] = eff_ps[-2]

        err_ss = np.zeros_like(tp)
        err_ps = np.zeros_like(tp)
        for i,t in enumerate(tp):
            err_ss[i] = np.sqrt(np.dot(\
                d_th_eff_ss_fun(t,dt),np.dot(cov_ss,d_th_eff_ss_fun(t,dt))))
            err_ps[i] = np.sqrt(np.dot(\
                d_th_eff_ps_fun(t,dt),np.dot(cov_ps,d_th_eff_ps_fun(t,dt))))
    elif corr in ['gA','gV']:
        fit_ss = fit_fh.fh_derivative(tp,dt,**parr_ss)
        eff_ss = fit_ss
        eff_ss[-1] = eff_ss[-2]
        fit_ps = fit_fh.fh_derivative(tp,dt,**parr_ps)
        eff_ps = fit_ps
        eff_ps[-1] = eff_ps[-2]

        err_ss = np.zeros_like(tp)
        err_ps = np.zeros_like(tp)
        for i,t in enumerate(tp):
            err_ss[i] = np.sqrt(np.dot(\
                th_grad_ss_fun(t,dt),np.dot(cov_ss,th_grad_ss_fun(t,dt))))
            err_ps[i] = np.sqrt(np.dot(\
                th_grad_ps_fun(t,dt),np.dot(cov_ps,th_grad_ps_fun(t,dt))))

    ax.fill_between(tp,eff_ss-err_ss,eff_ss+err_ss,color=clrs[0],alpha=.3)
    ax.fill_between(tp,eff_ps-err_ps,eff_ps+err_ps,color=clrs[1],alpha=.3)

if __name__ == "__main__":
    plt.ion()

    min_fh = fit('a09m310',params)

    print('plotting two point fit')
    l_ss = ['E_0','dE_10','zs_0','zs_1']
    l_ps = ['E_0','dE_10','zs_0','zp_0','zs_1','zp_1']
    plot_results('a09m310',params,min_fh,'proton',[l_ss,l_ps],'two_pt')

    print('plotting gA fit')
    l_ss = ['E_0','dE_10','zs_0','zs_1','gA_00','gA_11','gA_10','dAss_0','dAss_1']
    l_ps = ['E_0','dE_10','zs_0','zp_0','zs_1','zp_1',\
        'gA_00','gA_11','gA_10','dAps_0','dAps_1']
    plot_results('a09m310',params,min_fh,'gA',[l_ss,l_ps],'gA')

    print('plotting gV fit')
    l_ss = ['E_0','dE_10','zs_0','zs_1','gV_00','gV_11','gV_10','dVss_0','dVss_1']
    l_ps = ['E_0','dE_10','zs_0','zp_0','zs_1','zp_1',\
        'gV_00','gV_11','gV_10','dVps_0','dVps_1']
    plot_results('a09m310',params,min_fh,'gV',[l_ss,l_ps],'gV')

    plt.ioff()
    if run_from_ipython():
        plt.show(block=False)
    else:
        plt.show()
