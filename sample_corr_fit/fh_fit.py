from __future__ import print_function

import matplotlib.pyplot as plt
import tables as h5
import scipy as sp
import numpy as np
np.set_printoptions(linewidth=180)
import fit_functions as fit_fh
import iminuit as mn
import random
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
    'gA_11' :1.55,
    'gA_10' :-.01,
    'dAss_0':-4.7e-10,
    'dAps_0':-6.2e-8,
    'dAss_1':-4.2e-10,
    'dAps_1':1.5e-8,
    'gV_00' :1.02,
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
        self.nt_2p = self.t_p_f+1  - self.t_p_i
        self.nt_gA = self.t_gA_f+1 - self.t_gA_i
        self.nt_gV = self.t_gV_f+1 - self.t_gV_i
        self.t_2p  = np.arange(self.t_p_i,self.t_p_f+1)
        self.t_gA  = np.arange(self.t_gA_i,self.t_gA_f+1)
        self.t_gV  = np.arange(self.t_gV_i,self.t_gV_f+1)
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
        f[0:self.nt_2p] = fit_fh.c2pt(self.t_2p,E_0,zs_0,zs_0,\
            dE_10=dE_10,snk_1=zs_1,src_1=zs_1)
        # 2pt PS
        f[self.nt_2p:2*self.nt_2p] = fit_fh.c2pt(self.t_2p,E_0,zp_0,zs_0,\
            dE_10=dE_10,snk_1=zp_1,src_1=zs_1)
        # gA SS
        f[2*self.nt_2p:2*self.nt_2p+self.nt_gA] = \
            fit_fh.fh_derivative(self.t_gA,self.tau,E_0,zs_0,zs_0,gA_00,dAss_0,\
                dE_10=dE_10, snk_1=zs_1, src_1=zs_1, g_11=gA_11, g_10=gA_10,d_1=dAss_1)
        # gA PS
        f[2*self.nt_2p+self.nt_gA:2*(self.nt_2p+self.nt_gA)] = \
            fit_fh.fh_derivative(self.t_gA,self.tau,E_0,zp_0,zs_0,gA_00,dAps_0,\
                dE_10=dE_10, snk_1=zp_1, src_1=zs_1, g_11=gA_11, g_10=gA_10,d_1=dAps_1)
        # gV SS
        f[2*(self.nt_2p+self.nt_gA):2*(self.nt_2p+self.nt_gA)+self.nt_gV] = \
            fit_fh.fh_derivative(self.t_gV,self.tau,E_0,zs_0,zs_0,gV_00,dVss_0,\
                dE_10=dE_10, snk_1=zs_1, src_1=zs_1, g_11=gV_11, g_10=gV_10,d_1=dVss_1)
        # gV PS
        f[2*(self.nt_2p+self.nt_gA)+self.nt_gV:2*(self.nt_2p+self.nt_gA+self.nt_gV)] = \
            fit_fh.fh_derivative(self.t_gV,self.tau,E_0,zp_0,zs_0,gV_00,dVps_0,\
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

    return min_fh

if __name__ == "__main__":
    plt.ion()

    min_fh = fit('a09m310',params)

    plt.ioff()
    if run_from_ipython():
        plt.show(block=False)
    else:
        plt.show()
