import numpy as np
import scipy.special as spsp

def bs_corr(corr,Nbs,Mbs,seed=None):
    corr_bs = np.zeros(tuple([Nbs]) + corr.shape[1:],dtype=corr.dtype)
    np.random.seed(seed) # if None - it does not seed - I checked 14 May 2013
    # make bs_lst of shape (Nbs,Mbs)
    bs_lst = np.random.randint(0,corr.shape[0],(Nbs,Mbs))
    # use bs_lst to make corr_bs entries
    for bs in range(Nbs):
        corr_bs[bs] = corr[bs_lst[bs]].mean(axis=0)
    return corr_bs

def p_val(chisq,dof):
    return spsp.gammaincc(0.5*dof,0.5*chisq)

def dmdl(c0,cl,tau=1,t_col=1):
    r = cl / c0
    return 1./tau * (np.roll(r,-tau,axis=t_col) - r)

def c2pt(t,E_0,snk_0,src_0,dE_10=0.,snk_1=0.,src_1=0.,dE_21=0.,snk_2=0.,src_2=0.,**kwargs):
    '''
    Fit two point correlation function up to 3 states
    '''
    c  = snk_0 * src_0 * np.exp(-t * E_0)
    c += snk_1 * src_1 * np.exp(-t * (E_0 + dE_10))
    c += snk_2 * src_2 * np.exp(-t * (E_0 + dE_10 + dE_21))
    return c

def fh_numerator(t, E_0, snk_0, src_0, g_00, d_0,\
    dE_10=0., snk_1=0., src_1=0., g_11=0., g_10=0.,d_1=0.):
    '''
    FH numerator function currently supports up to 2 states
    '''
    num = ( (t-1)*g_00 *snk_0 *src_0  +d_0 ) *np.exp(-t * E_0)
    if snk_1 != 0.: #add 1st excited states
        num += ( (t-1) *g_11 *snk_1 *src_1 + d_1 ) *np.exp(-t * (E_0 + dE_10))
        ''' define R01 '''
        R01  = np.exp(-t * E_0) * np.exp(-dE_10 / 2)
        R01 += -np.exp(-t * (E_0 + dE_10)) *np.exp(dE_10 / 2)
        R01  = R01 / (np.exp(dE_10 / 2) - np.exp(-dE_10 / 2))
        num += g_10 *snk_0 *src_1 *R01

        ''' define R10 '''
        R10  = np.exp(-t * (E_0 + dE_10)) * np.exp(dE_10 / 2)
        R10 += -np.exp(-t * E_0) *np.exp(-dE_10 / 2)
        R10  = R10 / (np.exp(-dE_10 / 2) - np.exp(dE_10 / 2))
        num += g_10 *snk_1 * src_0 * R10
    return num

def fh_ratio(t, E_0, snk_0, src_0, g_00, d_0,\
    dE_10=0., snk_1=0., src_1=0., g_11=0., g_10=0.,d_1=0.):
    '''
    Construct the ratio FH/2pt function used in dm/dlam
    fh_ratio(t) = fh_numerator(t) / proton(t)
    '''
    R = fh_numerator(t, E_0, snk_0, src_0, g_00, d_0,\
        dE_10=dE_10, snk_1=snk_1, src_1=src_1, g_11=g_11, g_10=g_10,d_1=d_1)
    R = R / c2pt(t, E_0, snk_0, src_0, dE_10=dE_10, snk_1=snk_1, src_1=src_1)
    return R

def fh_derivative(t, tau, E_0, snk_0, src_0, g_00, d_0,\
    dE_10=0., snk_1=0., src_1=0., g_11=0., g_10=0.,d_1=0.):
    me  =  fh_ratio(t+tau, E_0, snk_0, src_0, g_00, d_0,\
        dE_10=dE_10, snk_1=snk_1, src_1=src_1, g_11=g_11, g_10=g_10,d_1=d_1)
    me += -fh_ratio(t    , E_0, snk_0, src_0, g_00, d_0,\
        dE_10=dE_10, snk_1=snk_1, src_1=src_1, g_11=g_11, g_10=g_10,d_1=d_1)
    return me / tau

''' NOW Define Theano variables to take derivatives '''
