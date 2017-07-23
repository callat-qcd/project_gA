import gvar as gv
import scipy.special as spsp
import numpy as np
import lsqfit

def format_data(switches, gadf, hqdf):
    gar_list = []
    epi_list = []
    aw0_list = []
    afs_list = []
    mpl_list = []
    for ens in switches['ensembles']:
        gar = gadf.query("ensemble=='%s'" %ens).sort_values(by='nbs')['ga'].as_matrix()
        epi = gadf.query("ensemble=='%s'" %ens).sort_values(by='nbs')['epi'].as_matrix()
        mpl = gadf.query("ensemble=='%s' and nbs==0" %ens)['mpil'].as_matrix()[0]
        awm = hqdf.query("ensemble=='%s'" %ens)['aw0_mean'].as_matrix()[0]
        aws = hqdf.query("ensemble=='%s'" %ens)['aw0_sdev'].as_matrix()[0]
        afs = hqdf.query("ensemble=='%s'" %ens)['alfs'].as_matrix()[0]
        d = gv.dataset.avg_data({'gar': gar, 'epi':epi}, bstrap=True)
        gar_list.append(d['gar'])
        epi_list.append(d['epi'])
        mpl_list.append(mpl)
        aw0_list.append(gv.gvar(awm,aws))
        afs_list.append(afs)
    data = {'y':{'gar': gar_list}, 'prior':{'epi': epi_list, 'aw0': aw0_list}, 'x':{'afs': afs_list}, 'mpl': mpl_list}
    return data

class fit_class():
    def __init__(self,ansatz,truncate,xsb,alpha,mL,FV):
        self.ansatz = ansatz
        self.n = truncate
        self.xsb = xsb
        self.alpha = alpha
        self.FV = FV
        # FV Bessel functions
        cn = np.array([6,12,8,6,24,24,0,12,30,24,24,8,24,48,0,6,48,36,24,24]) # |n| multiplicity
        mLn = [i*np.sqrt(np.arange(1,len(cn)+1)) for i in mL]
        kn0 = spsp.kn(0, mLn)
        kn1 = spsp.kn(1, mLn)
        self.F1 = np.array([np.sum(cn*kn0[i]-cn*kn1[i]/mLn[i]) for i in range(len(mL))])
        self.F3 = -1.5*np.array([np.sum(cn*kn1[i]/mLn[i]) for i in range(len(mL))])
        return None
    def get_priors(self,p,prior):
        for k in p[self.ansatz].keys():
            if int(k[1:]) <= self.n:
                prior[k] = p[self.ansatz][k] 
            else: pass
        return prior
    def dfv(self,p):
        r = 8./3.*p['epi']**2*(p['g0']**3*self.F1+p['g0']*self.F3)
        return r
    def fit_function(self,x,p):
        if self.ansatz == 'xpt':
            r = p['g0'] #*np.ones_like(p['epi']) # lo
            if self.n >= 1: # DWF O(a) discretization
                if self.xsb:
                    r += p['a1']*p['aw0']
            if self.n >= 2: # nlo
                r += -1.*p['epi']**2*(p['g0']+2.*p['g0']**3)*np.log(p['epi']**2) # nlo log
                r += p['epi']**2*p['c2'] # nlo counter term
                r += (p['aw0']**2/(4.*np.pi))*p['a2'] # nlo discretization
                if self.alpha:
                    r += x['afs']*(p['aw0']/(4.*np.pi))**2*p['s2'] # nlo alpha_s a^2
                if self.FV:
                    r += self.dfv(p)
            if self.n >= 3: # nnlo
                r += p['g0']*p['c3']*p['epi']**3 # nnlo log
            if self.n >= 4: # nnnlo analytic terms
                r += p['epi']**4*p['c4'] # nnnlo epi^4
                r += p['epi']**2*(p['aw0']**2/(4.*np.pi))*p['b4'] # nnnlo epi^2 a^2
                r += (p['aw0']**4/(4.*np.pi)**2)*p['a4'] # nnnlo a^4
            return r
        elif self.ansatz == 'taylor':
            r = p['c0']
            if self.n >= 2:
                r += p['c2']*p['epi']**2
                r += p['a2']*(p['aw0']**2/(4.*np.pi))
            if self.n >= 4:
                r += p['c4']*p['epi']**4
                r += p['a4']*(p['aw0']**4/(4.*np.pi)**2)
                r += p['b4']*p['epi']**2*(p['aw0']**2/(4.*np.pi))
            return r
        else:
            print('need to define fit function')
            raise SystemExit

def fit_data(s,p,data,phys):
    x = data['x']
    y = data['y']['gar']
    ansatz = s['ansatz']['type']
    truncate = s['ansatz']['truncation']
    xsb = s['ansatz']['xsb']
    alpha = s['ansatz']['alpha']
    FV = s['ansatz']['FV']
    fitc = fit_class(ansatz,truncate,xsb,alpha,data['mpl'],FV)
    prior = fitc.get_priors(p,data['prior'])
    fit = lsqfit.nonlinear_fit(data=(x,y),prior=prior,fcn=fitc.fit_function)
    print(fit)
    phys = eval_phys(phys,fitc,fit)
    return {'fit':fit, 'phys':phys, 'fitc': fitc}

def eval_phys(phys,fitc,fit):
    x = {'afs': 0}
    F = phys['fpi']/np.sqrt(2)
    m = phys['mpi']
    epi = m/(4.*np.pi*F)
    priorc = dict()
    for k in fit.p.keys():
        if k == 'epi':
            priorc[k] = epi
        elif k == 'aw0':
            priorc[k] = 0
        else:
            priorc[k] = fit.p[k]
    fitc.FV = False
    phys = fitc.fit_function(x,priorc)
    print(phys)
    return {'result': phys, 'priorc': priorc, 'epi': epi}

if __name__=='__main__':
    print("chipt library")
