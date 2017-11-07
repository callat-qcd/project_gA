import sys
import gvar as gv
import scipy.special as spsp
import scipy.stats as stats
import numpy as np
import lsqfit
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

def format_data(switches, gadf, hqdf):
    gar_list = []
    epi_list = []
    aw0_list = []
    afs_list = []
    mpl_list = []
    ed_list  = []
    for ens in switches['ensembles']:
        gar = gadf.query("ensemble=='%s'" %ens).sort_values(by='nbs')['ga'].as_matrix()
        epi = gadf.query("ensemble=='%s'" %ens).sort_values(by='nbs')['epi'].as_matrix()
        mpl = gadf.query("ensemble=='%s' and nbs==0" %ens)['mpil'].as_matrix()[0]
        awm = hqdf.query("ensemble=='%s'" %ens)['aw0_mean'].as_matrix()[0]
        aws = hqdf.query("ensemble=='%s'" %ens)['aw0_sdev'].as_matrix()[0]
        afs = hqdf.query("ensemble=='%s'" %ens)['alfs'].as_matrix()[0]
        ed  = hqdf.query("ensemble=='%s'" %ens)['eps_delta'].as_matrix()[0]
        d = gv.dataset.avg_data({'gar': gar, 'epi':epi}, bstrap=True)
        gar_list.append(d['gar'])
        epi_list.append(d['epi'])
        mpl_list.append(mpl)
        aw0_list.append(gv.gvar(awm,aws))
        ed_list.append(gv.gvar(ed,switches['eps_delta_sig']*ed))
        afs_list.append(afs)
    data = {'y':{'gar': gar_list}, 'prior':{'epi': epi_list, 'aw0': aw0_list,'ed':ed_list},
            'x':{'afs': afs_list}, 'mpl': mpl_list}
    return data

class fit_class():
    def __init__(self,ansatz,truncate,xsb,alpha,mL,FV):
        self.ansatz = ansatz
        self.n = truncate
        self.xsb = xsb
        self.alpha = alpha
        self.FV = FV
        self.at = '%s_%s' %(ansatz,truncate)
        # FV Bessel functions
        cn = np.array([6,12,8,6,24,24,0,12,30,24,24,8,24,48,0,6,48,36,24,24]) # |n| multiplicity
        mLn = [i*np.sqrt(np.arange(1,len(cn)+1)) for i in mL]
        kn0 = spsp.kn(0, mLn)
        kn1 = spsp.kn(1, mLn)
        self.F1 = np.array([np.sum(cn*kn0[i]-cn*kn1[i]/mLn[i]) for i in range(len(mL))])
        self.F3 = -1.5*np.array([np.sum(cn*kn1[i]/mLn[i]) for i in range(len(mL))])
        return None
    def get_priors(self,p,prior):
        a = self.ansatz.split('-')[0]
        for k in p[a].keys():
            if int(k[-1]) <= self.n:
                mean = p[a][k].mean
                sdev = p[a][k].sdev
                prior['%s_%s' %(self.at,k)] = gv.gvar(mean,sdev)
            else: pass
        return prior
    def dfv(self,p):
        epi = p['epi']
        g0 = p['%s_g0' %self.at]
        r = 8./3.*epi**2*(g0**3*self.F1+g0*self.F3)
        return r
    def R(self,zz):
        try:
            d = len(zz)
        except:
            d = 0
            zz = [zz]
        r = np.zeros_like(zz)
        for i,z in enumerate(zz):
            if z == 0:
                r[i]  = 0
            elif z > 0. and z < 1.:
                r[i]  = np.sqrt(1-z) * np.log((1-np.sqrt(1-z))/(1+np.sqrt(1-z)))
                r[i] += np.log(4./z)
            elif z == 1:
                r[i]  = np.log(4.)
            elif z > 1.:
                r[i]  = 2*np.sqrt(z-1)*np.arctan(z) + np.log(4./z)
            else:
                print('R(z) only defined for z > 0')
                sys.exit(-1)
        if d == 0:
            r = r[0]
        return r
    def fit_function(self,x,p):
        def nnnlo_analytic_xpt(x,p):
            epi = p['epi']
            aw0 = p['aw0']
            g0 = p['%s_g0' %self.at]
            r = g0 #*np.ones_like(p['epi']) # lo
            if self.n >= 1: # DWF O(a) discretization
                if self.xsb:
                    a1 = p['%s_a1' %self.at]
                    r += a1*aw0
            if self.n >= 2: # nlo
                c2 = p['%s_c2' %self.at]
                a2 = p['%s_a2' %self.at]
                g2  = g0 +2.*g0**3 # nucleon terms
                r += -1.*epi**2 * g2 *np.log(epi**2) # nlo log
                # counter terms
                r += epi**2*c2 # nlo counter term
                r += (aw0**2/(4.*np.pi))*a2 # nlo discretization
                if self.alpha:
                    s2 = p['%s_s2' %self.at]
                    r += x['afs']*(aw0/(4.*np.pi))**2*s2 # nlo alpha_s a^2
                if self.FV:
                    r += self.dfv(p)
            if self.n >= 3: # nnlo
                c3 = p['%s_c3' %self.at]
                r += g0*c3*epi**3 # nnlo log
            if self.n >= 4: # nnnlo analytic terms
                c4 = p['%s_c4' %self.at]
                b4 = p['%s_b4' %self.at]
                a4 = p['%s_a4' %self.at]
                r += epi**4*c4 # nnnlo epi^4
                r += epi**2*(aw0**2/(4.*np.pi))*b4 # nnnlo epi^2 a^2
                r += (aw0**4/(4.*np.pi)**2)*a4 # nnnlo a^4
            return r
        def nnnlo_log2_xpt(x,p):
            r = 0
            if self.n >= 4:
                epi = p['epi']
                g0 = p['%s_g0' %self.at]
                l2 = -16./3*g0 -11./3*g0**3 +16.*g0**5
                l2 += 4.*(2*g0 + 4**g0**3)
                r = l2/4.*epi**4 * (np.log(epi**2))**2
            return r
        def nnnlo_log_xpt(x,p):
            r = 0
            if self.n >= 4:
                epi = p['epi']
                gm4 = p['%s_gm4' %self.at]
                r = gm4*epi**4*np.log(epi**2)
            return r
        def nlo_delta_xpt(x,p):
            r = 0
            if self.n >= 2:
                epi = p['epi']
                ed = p['ed']
                g0 = p['%s_g0' %self.at]
                gnd0 = p['%s_gnd0' %self.at]
                gdd0 = p['%s_gdd0' %self.at]
                g2 = gnd0**2*(2.*g0/9 +50.*gdd0/81) #delta
                r  = -1.*g2*epi**2*np.log(epi**2)
                # extra delta terms
                g2r  = gnd0**2*epi**2 * 32.*g0 / 27
                g2r += gnd0**2*ed**2 * (76.*g0/27 +100.*gdd0/81)
                r   += -1.*g2r*self.R(epi**2 / ed**2)
                g2d  = 76.*g0*gnd0**2 / 27
                g2d += 100.*gdd0*gnd0**2 / 81
                r   += -1.*g2d*ed**2*np.log(4.*ed**2/epi**2)
                # delta mpi^3 term
                r   += 32.*np.pi/27*g0*gnd0**2*epi**3/ed
            return r
        if self.ansatz == 'xpt':
            r = nnnlo_analytic_xpt(x,p)
            return r
        elif self.ansatz == 'xpt-doublelog':
            r = nnnlo_analytic_xpt(x,p)
            r += nnnlo_log2_xpt(x,p)
            return r
        elif self.ansatz == 'xpt-full':
            r = nnnlo_analytic_xpt(x,p)
            r += nnnlo_log2_xpt(x,p)
            r += nnnlo_log_xpt(x,p)
            return r
        elif self.ansatz == 'xpt-delta':
            r = nnnlo_analytic_xpt(x,p)
            r += nlo_delta_xpt(x,p)
            return r
        elif self.ansatz == 'taylor':
            epi = p['epi']
            aw0 = p['aw0']
            c0 = p['%s_c0' %self.at]
            r = c0
            if self.n >= 2:
                c2 = p['%s_c2' %self.at]
                a2 = p['%s_a2' %self.at]
                r += c2*epi**2
                r += a2*(aw0**2/(4.*np.pi))
                if self.FV:
                    r += self.dfv(p)
            if self.n >= 4:
                c4 = p['%s_c4' %self.at]
                b4 = p['%s_b4' %self.at]
                a4 = p['%s_a4' %self.at]
                r += c4*epi**4
                r += a4*(aw0**4/(4.*np.pi)**2)
                r += b4*epi**2*(aw0**2/(4.*np.pi))
            return r
        elif self.ansatz == 'linear':
            epi = p['epi']
            aw0 = p['aw0']
            c0 = p['%s_c0' %self.at]
            r = c0
            if self.n >= 2:
                c2 = p['%s_c2' %self.at]
                a2 = p['%s_a2' %self.at]
                r += c2*epi
                r += a2*(aw0**2/(4.*np.pi))
                if self.FV:
                    r += self.dfv(p)
            if self.n >= 4:
                c4 = p['%s_c4' %self.at]
                a4 = p['%s_a4' %self.at]
                r += c4*epi**2
                r += a4*(aw0**4/(4.*np.pi)**2)
            return r
        elif self.ansatz == 'constant':
            epi = p['epi']
            aw0 = p['aw0']
            c0 = p['%s_c0' %self.at]
            r = c0
            if self.n >= 2:
                a2 = p['%s_a2' %self.at]
                r += a2*(aw0**2/(4.*np.pi))
                if self.FV:
                    r += self.dfv(p)
            if self.n >= 4:
                a4 = p['%s_a4' %self.at]
                r += a4*(aw0**4/(4.*np.pi)**2)
            return r
        else:
            print('need to define fit function')
            raise SystemExit

def fit_data(s,p,data,phys):
    x = data['x']
    y = data['y']['gar']
    ansatz_list = s['ansatz']['type']
    xsb = s['ansatz']['xsb']
    alpha = s['ansatz']['alpha']
    FV = s['ansatz']['FV']
    result = dict()
    for ansatz_truncate in ansatz_list:
        ansatz = ansatz_truncate.split('_')[0]
        truncate = int(ansatz_truncate.split('_')[1])
        fitc = fit_class(ansatz,truncate,xsb,alpha,data['mpl'],FV)
        prior = fitc.get_priors(p,data['prior'])
    for ansatz_truncate in ansatz_list:
        ansatz = ansatz_truncate.split('_')[0]
        truncate = int(ansatz_truncate.split('_')[1])
        fitc = fit_class(ansatz,truncate,xsb,alpha,data['mpl'],FV)
        fit = lsqfit.nonlinear_fit(data=(x,y),prior=prior,fcn=fitc.fit_function)
        phys_pt = eval_phys(phys,fitc,fit)
        result[ansatz_truncate] = {'fit':fit, 'phys':phys_pt, 'fitc': fitc}
    return result

def eval_phys(phys,fitc,fit):
    x = {'afs': 0}
    F = phys['fpi']/np.sqrt(2)
    m = phys['mpi']
    epi = m/(4.*np.pi*F)
    ed = phys['Delta']/(4.*np.pi*F)
    priorc = dict()
    for k in fit.p.keys():
        if k == 'epi':
            priorc[k] = epi
        elif k == 'aw0':
            priorc[k] = 0
        elif k == 'ed':
            priorc[k] = np.array(ed)
        else:
            priorc[k] = fit.p[k]
    fitc.FV = False
    phys = fitc.fit_function(x,priorc)
    return {'result': phys, 'priorc': priorc, 'epi': epi}

def error_budget(s,result_list):
    err = dict()
    for ansatz_truncate in s['ansatz']['type']:
        result = result_list[ansatz_truncate]
        fit = result['fit']
        prior = fit.prior
        priorc = result['phys']['priorc']
        phys = result['phys']['result']
        statistical = phys.partialsdev(fit.y)
        inputerror = phys.partialsdev(priorc['epi'],priorc['ed'])
        # compile chiral and discretization lists then splat as function input
        X_list = []
        d_list = []
        k_list = []
        at = ansatz_truncate.split('_')
        ansatz = at[0]
        n = int(at[1])
        for key in prior.keys():
            ks = key.split('_')
            k = ks[-1]
            if k[0] in ['c','b','g'] and ansatz_truncate in key:
                X_list.append(prior[key])
                k_list.append(key)
            if k[0] in ['a','s'] and ansatz_truncate in key:
                d_list.append(prior[key])
                k_list.append(key)
        chiral      = phys.partialsdev(*X_list)
        disc        = phys.partialsdev(*d_list)
        pct = {'stat':[statistical/phys.mean*100],'chiral':[chiral/phys.mean*100],'disc':[disc/phys.mean*100],'input':[inputerror/phys.mean*100],'total':[phys.sdev/phys.mean*100]}
        std = {'stat':statistical,'chiral':chiral,'disc':disc,'input':inputerror,'total':phys.sdev}
        err[ansatz_truncate] = {'pct':pct,'std':std,'mean':phys.mean}
    return err

def bma(switches,result,fverror,isospin):
    # read Bayes Factors
    logGBF_list = []
    for a in switches['ansatz']['type']:
        logGBF_list.append(result[a]['fit'].logGBF)
    # initiate a bunch of parameters
    # gA
    gA = 0
    gA_lst = []
    gA_dict = dict()
    # weights
    w_lst = []
    wd = dict()
    # p. dist. fcn
    pdf = 0
    pdfdict = dict()
    # c. dist. fcn.
    cdf = 0
    cdfdict = dict()
    # for plotting
    x = np.linspace(1.222,1.352,13000)
    for a in switches['ansatz']['type']:
        r = result[a]['phys']['result']
        gA_dict[a] = r
        w = 1/sum(np.exp(np.array(logGBF_list)-result[a]['fit'].logGBF))
        wd[a] = w
        w_lst.append(w)
        gA += w*r
        gA_lst.append(r.mean)
        p = stats.norm.pdf(x,r.mean,r.sdev)
        pdf += w*p
        pdfdict[a] = w*p
        c = stats.norm.cdf(x,r.mean,r.sdev)
        cdf += w*c
        cdfdict[a] = w*c
    gA_lst = np.array(gA_lst)
    w_lst = np.array(w_lst)
    model_var = np.sum(w_lst*(gA_lst**2 - gA.mean**2))
    final_error = np.sqrt(gA.sdev**2 + fverror**2 + isospin**2)
    additional_var = {'fv':fverror**2, 'isospin':isospin**2, 'model': model_var,'total':final_error**2}
    model_budget = modelavg_error(switches,result,gA,additional_var)
    error = {'E(gA)': gA.mean, 's(gA)': final_error, 's(Mk)': np.sqrt(model_var), 'weights': wd, 'error_budget': model_budget, 'gA_dict':gA_dict}
    plot_params = {'x':x, 'pdf':pdf, 'pdfdict':pdfdict, 'cdf':cdf, 'cdfdict':cdfdict}
    return error, plot_params

def modelavg_error(s,result_list,ga,var):
    result = result_list[s['ansatz']['type'][-1]]
    fit = result['fit']
    prior = fit.prior
    phys_epi = result['phys']['priorc']['epi']
    phys_ed = result['phys']['priorc']['ed']
    statistical = ga.partialsdev(fit.y,phys_epi,phys_ed)
    X_list = []
    d_list = []
    k_list = []
    for key in prior.keys():
        ks = key.split('_')
        k = ks[-1]
        if k[0] in ['c','b','g']:
            X_list.append(prior[key])
            k_list.append(key)
        if k[0] in ['a','s'] and key not in ['aw0']:
            d_list.append(prior[key])
            k_list.append(key)
    chiral      = ga.partialsdev(*X_list)
    disc        = ga.partialsdev(*d_list)
    total = np.sqrt(ga.sdev**2+var['fv']+var['isospin']+var['model'])
    pct = {'stat':[statistical/ga.mean*100],'chiral':[chiral/ga.mean*100],'disc':[disc/ga.mean*100],'FV':[np.sqrt(var['fv'])/ga.mean*100],'isospin':[np.sqrt(var['isospin'])/ga.mean*100],'model':[np.sqrt(var['model'])/ga.mean*100],'total':[total/ga.mean*100]}
    std = {'stat':statistical,'chiral':chiral,'disc':disc,'fv':np.sqrt(var['fv']),'isospin':np.sqrt(var['isospin']),'total_fit':ga.sdev,'total':total}
    return {'pct':pct,'std':std,'mean':ga.mean}

class plot_chiral_fit():
    def __init__(self):
        self.loc = './plots'
        self.plot_params = dict()
        self.plot_params['l1648f211b580m0217m065m838']  = {'abbr': 'a15m400',  'color': '#ec5d57', 'marker': 'h', 'label': ''}
        self.plot_params['l1648f211b580m0166m065m838']  = {'abbr': 'a15m350',  'color': '#ec5d57', 'marker': 'p', 'label': ''}
        self.plot_params['l1648f211b580m013m065m838']    = {'abbr': 'a15m310',  'color': '#ec5d57', 'marker': 's', 'label': '$a\simeq 0.15$~fm'}
        self.plot_params['l2448f211b580m0064m0640m828']  = {'abbr': 'a15m220',  'color': '#ec5d57', 'marker': '^', 'label': ''}
        self.plot_params['l3248f211b580m00235m0647m831'] = {'abbr': 'a15m130',  'color': '#ec5d57', 'marker': 'o', 'label': ''}
        self.plot_params['l2464f211b600m0170m0509m635']  = {'abbr': 'a12m400',  'color': '#70bf41', 'marker': 'h', 'label': ''}
        self.plot_params['l2464f211b600m0130m0509m635']  = {'abbr': 'a12m350',  'color': '#70bf41', 'marker': 'p', 'label': ''}
        self.plot_params['l2464f211b600m0102m0509m635']  = {'abbr': 'a12m310',  'color': '#70bf41', 'marker': 's', 'label': '$a\simeq 0.12$~fm'}
        self.plot_params['l2464f211b600m00507m0507m628'] = {'abbr': 'a12m220S', 'color': '#70bf41', 'marker': '^', 'label': ''}
        self.plot_params['l3264f211b600m00507m0507m628'] = {'abbr': 'a12m220',  'color': '#70bf41', 'marker': '^', 'label': ''}
        self.plot_params['l4064f211b600m00507m0507m628'] = {'abbr': 'a12m220L', 'color': '#70bf41', 'marker': '^', 'label': ''}
        self.plot_params['l4864f211b600m00184m0507m628'] = {'abbr': 'a12m130',  'color': '#70bf41', 'marker': 'o', 'label': ''}
        self.plot_params['l3264f211b630m0124m037m440']  = {'abbr': 'a09m400',  'color': '#51a7f9', 'marker': 'h', 'label': ''}
        self.plot_params['l3264f211b630m00945m037m440']  = {'abbr': 'a09m350',  'color': '#51a7f9', 'marker': 'p', 'label': ''}
        self.plot_params['l3296f211b630m0074m037m440']   = {'abbr': 'a09m310',  'color': '#51a7f9', 'marker': 's', 'label': '$a\simeq 0.09$~fm'}
        self.plot_params['l4896f211b630m00363m0363m430'] = {'abbr': 'a09m220',  'color': '#51a7f9', 'marker': '^', 'label': ''}
        self.title = {
            'xpt_4':r'NNLO+ct $\chi$PT','xpt_3':r'NNLO $\chi$PT',
            'xpt_2':r'NLO $\chi$PT',
            'xpt-full_4':r'N3LO $\chi$PT',
            'taylor_2':r'NLO Taylor $\epsilon_\pi^2$','taylor_4':r'NNLO Taylor $\epsilon_\pi^2$',
            'linear_2':r'NLO Taylor $\epsilon_\pi$','linear_4':r'NNLO Taylor $\epsilon_\pi$'
            }
    def plot_chiral(self,s,data,result_list):
        # convergence
        def plot_convergence(result,xp,ansatz):
            fitc = result['fitc']
            init_order = fitc.n
            x = xp['x']
            priorx = xp['priorx']
            if ansatz in ['taylor','linear']:
                tn = fitc.n//2+1
                order = np.zeros(tn)
                order[0] = 1
                for i in range(1,tn):
                    order[i] = 2*i
            else:
                tn = fitc.n
                order = range(1,tn+1)
            ls_list = ['-','--','-.',':']
            label = ['LO','NLO','NNLO','N3LO']
            phys_converge = []
            for n in range(tn):
                fitc.n = order[n]
                extrap = fitc.fit_function(x,priorx)
                # print numerical breakdown
                converge_prior = dict(priorx)
                converge_prior['epi'] = result['phys']['epi']
                phys_converge.append(fitc.fit_function(x,converge_prior))
                if n == 0:
                    extrap = [extrap for i in range(len(priorx['epi']))]
                mean = np.array([i.mean for i in extrap])
                sdev = np.array([i.sdev for i in extrap])
                ax.fill_between(priorx['epi'],mean+sdev,mean-sdev,alpha=0.4,label=label[n])
            fitc.n = init_order
            return ax
        # chiral extrapolation
        def c_chiral(ax,result):
            fit = result['fit']
            fitc = result['fitc']
            epi_extrap = np.linspace(0.0001,0.3501,3501)
            aw0_list = [gv.gvar(0.8804,0.003), gv.gvar(0.7036,0.005), gv.gvar(0.5105,0.003)]
            afs_list = [0.58801,0.53796,0.43356]
            pp = self.plot_params
            color_list = [pp['l1648f211b580m013m065m838']['color'], pp['l2464f211b600m0170m0509m635']['color'], pp['l3296f211b630m0074m037m440']['color']]
            label = ['$g_A(\epsilon_\pi,a\simeq 0.15$~fm$)$','$g_A(\epsilon_\pi,a\simeq 0.12$~fm$)$','$g_A(\epsilon_\pi,a\simeq 0.09$~fm$)$']
            fitc.FV = False
            ra = dict()
            for i in range(len(aw0_list)):
                x = {'afs': afs_list[i]}
                priorx = dict()
                for k in fit.p.keys():
                    if k == 'epi':
                        priorx[k] = epi_extrap
                    elif k == 'aw0':
                        priorx[k] = aw0_list[i]
                    else:
                        priorx[k] = fit.p[k]
                extrap = fitc.fit_function(x,priorx)
                ax.errorbar(x=epi_extrap,y=[j.mean for j in extrap],ls='-',marker='',elinewidth=1,color=color_list[i],label=label[i])
                ra[i] = extrap
            return ax, ra
        def c_continuum(ax,result):
            fit = result['fit']
            fitc = result['fitc']
            epi_extrap = np.linspace(0.0001,0.3501,3501)
            fitc.FV = False
            x = {'afs': 0}
            priorx = dict()
            for k in fit.p.keys():
                if k == 'epi':
                    priorx[k] = epi_extrap
                elif k == 'aw0':
                    priorx[k] = 0
                else:
                    priorx[k] = fit.p[k]
            extrap = fitc.fit_function(x,priorx)
            mean = np.array([j.mean for j in extrap])
            sdev = np.array([j.sdev for j in extrap])
            epi_phys = result['phys']['epi']
            ax.axvspan(epi_phys.mean-epi_phys.sdev, epi_phys.mean+epi_phys.sdev, alpha=0.4, color='#a6aaa9')
            ax.axvline(epi_phys.mean,ls='--',color='#a6aaa9')
            ax.fill_between(epi_extrap,mean+sdev,mean-sdev,alpha=0.4,color='#b36ae2',label='$g_A^{LQCD}(\epsilon_\pi,a=0)$')
            ax.errorbar(x=epi_extrap,y=mean,ls='--',marker='',elinewidth=1,color='#b36ae2')
            return ax, {'x':x, 'priorx':priorx}, {'epi':epi_extrap,'y':extrap}
        def c_data(ax,s,result):
            x = result['fit'].prior['epi']
            if s['ansatz']['FV']:
                y = result['fit'].y - result['fitc'].dfv(result['fit'].p)
            else:
                y = result['fit'].y
            datax = []
            datay = []
            elist = []
            for i,e in enumerate(s['ensembles']):
                ax.errorbar(x=x[i].mean,xerr=x[i].sdev,y=y[i].mean,yerr=y[i].sdev,ls='None',marker=self.plot_params[e]['marker'],fillstyle='full',markersize='5',elinewidth=1,capsize=2,color=self.plot_params[e]['color'],label=self.plot_params[e]['label'])
                datax.append(x[i])
                datay.append(y[i])
                elist.append(e)
            return ax, {'x':np.array(x),'y':np.array(y),'ens':np.array(elist)}
        def c_pdg(ax,result):
            gA_pdg = [1.2723, 0.0023]
            ax.errorbar(x=result['phys']['epi'].mean,y=gA_pdg[0],yerr=gA_pdg[1],ls='None',marker='o',fillstyle='none',markersize='8',capsize=2,color='black',label='$g_A^{PDG}=1.2723(23)$')
            return ax
        def c_legend(ax):
            handles, labels = ax.get_legend_handles_labels()
            l0 = [handles[0],handles[-1]]
            l1 = [handles[i] for i in range(len(handles)-2,0,-1)]
            leg = ax.legend(handles=l0,numpoints=1,loc=1,ncol=1,fontsize=12,edgecolor='k',fancybox=False)
            ax.legend(handles=l1,numpoints=1,loc=4,ncol=2,fontsize=12,edgecolor='k',fancybox=False)
            plt.gca().add_artist(leg)
            return None
        ### Chiral extrapolation
        r_chiral = dict()
        for ansatz_truncate in s['ansatz']['type']:
            if ansatz_truncate.split('_')[0] in ['xpt-delta']:
                print('CAN NOT PRINT: eps_delta(eps_pi) = unknown')
                continue
            result = result_list[ansatz_truncate]
            fig = plt.figure('%s chiral extrapolation' %ansatz_truncate,figsize=(7,4.326237))
            ax = plt.axes([0.15,0.15,0.8,0.8])
            # continuum extrapolation
            ax, xp, r0 = c_continuum(ax,result) # xp is used to make chipt convergence plot
            # plot chiral extrapolation
            ax, ra = c_chiral(ax,result)
            # plot data
            ax, rd = c_data(ax,s,result)
            r_chiral[ansatz_truncate] = {'r0':r0,'ra':ra,'rd':rd}
            # plot pdg
            ax = c_pdg(ax,result)
            # make legend
            c_legend(ax)
            # format plot
            ax.set_ylim([1.075,1.375])
            ax.set_xlim([0,0.32])
            ax.set_xlabel('$\epsilon_\pi=m_\pi/(4\pi F_\pi)$', fontsize=20)
            ax.set_ylabel('$g_A$', fontsize=20)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            ax.set_title(self.title[ansatz_truncate],fontdict={'fontsize':20,'verticalalignment':'top','horizontalalignment':'left'},x=0.05,y=0.9)
            self.ax = ax
            if s['save_figs']:
                plt.savefig('%s/chiral_%s.pdf' %(self.loc,ansatz_truncate),transparent=True)
            plt.draw()
            ### Convergence
            fig = plt.figure('%s chiral convergence' %ansatz_truncate,figsize=(7,4.326237))
            ax = plt.axes([0.15,0.15,0.8,0.8])
            ax = plot_convergence(result,xp,ansatz_truncate.split('_')[0])
            # plot physical pion point
            epi_phys = result['phys']['epi']
            ax.axvspan(epi_phys.mean-epi_phys.sdev, epi_phys.mean+epi_phys.sdev, alpha=0.4, color='#a6aaa9')
            ax.axvline(epi_phys.mean,ls='--',color='#a6aaa9')
            # make legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles,loc=3,ncol=2,fontsize=12,edgecolor='k',fancybox=False)
            # format plot
            ax.set_ylim([1.075,1.375])
            ax.set_xlim([0,0.32])
            ax.set_xlabel('$\epsilon_\pi=m_\pi/(4\pi F_\pi)$', fontsize=20)
            ax.set_ylabel('$g_A$', fontsize=20)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            ax.set_title(self.title[ansatz_truncate],fontdict={'fontsize':20,'verticalalignment':'top','horizontalalignment':'left'},x=0.05,y=0.9)
            if s['save_figs']:
                plt.savefig('%s/convergence_%s.pdf' %(self.loc,ansatz_truncate),transparent=True)
            plt.draw()
        return r_chiral
    def plot_continuum(self,s,data,result_list):
        def a_chiral(ax,result):
            fit = result['fit']
            fitc = result['fitc']
            epi_list = [0.1135, 0.182, 0.248, 0.2714, 0.29828]
            aw0_extrap = np.linspace(0.0,0.9001,9101)
            epi = 0
            c15 = self.plot_params['l1648f211b580m013m065m838']['color']
            c12 = self.plot_params['l2464f211b600m0170m0509m635']['color']
            c09 = self.plot_params['l3296f211b630m0074m037m440']['color']
            ls_list = ['-','--','-.',':','-']
            label = ['$g_A(\epsilon^{(130)}_\pi,\epsilon_a)$','$g_A(\epsilon^{(220)}_\pi,\epsilon_a)$','$g_A(\epsilon^{(310)}_\pi,\epsilon_a)$','$g_A(\epsilon^{(350)}_\pi,\epsilon_a)$','$g_A(\epsilon^{(400)}_\pi,\epsilon_a)$']
            color = ['black','black','black','black','black']
            dashes = [8, 4, 2, 4, 2, 4]
            fitc.FV = False
            rm = dict()
            for i in range(len(epi_list)):
                x = {'afs': 0}
                priorx = dict()
                for k in fit.p.keys():
                    if k == 'aw0':
                        priorx[k] = aw0_extrap
                    elif k == 'epi':
                        priorx[k] = epi_list[i]
                    else:
                        priorx[k] = fit.p[k]
                extrap = fitc.fit_function(x,priorx)
                aw0_extrap_plot = aw0_extrap**2/(4*np.pi)
                if i == 4:
                    ax.errorbar(x=aw0_extrap_plot,y=[j.mean for j in extrap],ls=ls_list[i],dashes=dashes,marker='',elinewidth=1,color=color[i],label=label[i])
                else:
                    ax.errorbar(x=aw0_extrap_plot,y=[j.mean for j in extrap],ls=ls_list[i],marker='',elinewidth=1,color=color[i],label=label[i])
                rm[i] = extrap
            return ax, rm
        def a_cont(ax,result):
            fit = result['fit']
            fitc = result['fitc']
            epi_phys = result['phys']['epi']
            aw0_extrap = np.linspace(0.0,0.9001,9101)
            fitc.FV = False
            x = {'afs': 0}
            priorx = dict()
            for k in fit.p.keys():
                if k == 'epi':
                    priorx[k] = epi_phys
                elif k == 'aw0':
                    priorx[k] = aw0_extrap
                else:
                    priorx[k] = fit.p[k]
            extrap = fitc.fit_function(x,priorx)
            mean = np.array([j.mean for j in extrap])
            sdev = np.array([j.sdev for j in extrap])
            aw0_extrap_plot = aw0_extrap**2/(4*np.pi)
            ax.fill_between(aw0_extrap_plot,mean+sdev,mean-sdev,alpha=0.4,color='#b36ae2',label='$g_A^{LQCD}(\epsilon_\pi^{phys.},\epsilon_a)$')
            ax.errorbar(x=aw0_extrap_plot,y=mean,ls='-',marker='',elinewidth=1,color='#b36ae2')
            return ax, {'x':x, 'priorx':priorx}, {'aw0_extrap_plot':aw0_extrap_plot,'y':extrap}
        def a_data(ax,s,result):
            x = result['fit'].prior['aw0']
            if s['ansatz']['FV']:
                y = result['fit'].y - result['fitc'].dfv(result['fit'].p)
            else:
                y = result['fit'].y
            xlist = []
            ylist = []
            elist = []
            for i,e in enumerate(s['ensembles']):
                xplot = x[i]**2/(4.*np.pi)
                ax.errorbar(x=xplot.mean,xerr=xplot.sdev,y=y[i].mean,yerr=y[i].sdev,ls='None',marker=self.plot_params[e]['marker'],fillstyle='full',markersize='5',elinewidth=1,capsize=2,color=self.plot_params[e]['color'])
                xlist.append(xplot)
                ylist.append(y[i])
                elist.append(e)
            return ax, {'x':np.array(xlist),'y':np.array(ylist),'ens':np.array(elist)}
        def a_pdg(ax,result):
            gA_pdg = [1.2723, 0.0023]
            ax.errorbar(x=0,y=gA_pdg[0],yerr=gA_pdg[1],ls='None',marker='o',fillstyle='none',markersize='8',capsize=2,color='black',label='$g_A^{PDG}=1.2723(23)$')
            return ax
        def a_legend(ax):
            handles, labels = ax.get_legend_handles_labels()
            l0 = [handles[0],handles[-1]]
            l1 = [handles[i] for i in range(len(handles)-2,0,-1)]
            leg = ax.legend(handles=l0,numpoints=1,loc=1,ncol=1,fontsize=12,edgecolor='k',fancybox=False)
            ax.legend(handles=l1,numpoints=1,loc=3,ncol=2,fontsize=12,edgecolor='k',fancybox=False)
            plt.gca().add_artist(leg)
            return None
        r_cont = dict()
        for ansatz_truncate in s['ansatz']['type']:
            if ansatz_truncate.split('_')[0] in ['xpt-delta']:
                print('CAN NOT PRINT: eps_delta(eps_pi) = unknown')
                continue
            result = result_list[ansatz_truncate]
            fig = plt.figure('%s continuum extrapolation' %ansatz_truncate,figsize=(7,4.326237))
            ax = plt.axes([0.15,0.15,0.8,0.8])
            # continuum extrapolation
            ax, res, r0 = a_cont(ax,result)
            # chiral extrapolation
            ax, rm = a_chiral(ax,result)
            # plot data
            ax, rd = a_data(ax,s,result)
            r_cont[ansatz_truncate] = {'r0':r0,'rm':rm,'rd':rd}
            # plot PDG
            ax = a_pdg(ax,result)
            # make legend
            a_legend(ax)
            # format plot
            ax.set_ylim([1.075,1.375])
            ax.set_xlim([-0.001,0.81/(4*np.pi)])
            ax.set_xlabel('$\epsilon_a^2=a^2/(4\pi w^2_0)$', fontsize=20)
            ax.set_ylabel('$g_A$', fontsize=20)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            ax.set_title(self.title[ansatz_truncate],fontdict={'fontsize':20,'verticalalignment':'top','horizontalalignment':'left'},x=0.05,y=0.9)
            if s['save_figs']:
                plt.savefig('%s/continuum_%s.pdf' %(self.loc,ansatz_truncate),transparent=True)
            plt.draw()
        return r_cont
    def plot_volume(self,s,data,result_list):
        if s['ansatz']['FV']:
            def v_vol(ax,s,result,ansatz_truncate):
                fit = result['fit']
                ansatz = ansatz_truncate.split('_')[0]
                truncate = int(ansatz_truncate.split('_')[1])
                xsb = s['ansatz']['xsb']
                alpha = s['ansatz']['alpha']
                FV = s['ansatz']['FV']
                mpiL_extrap = np.linspace(3,10,500)
                fitc = fit_class(ansatz,truncate,xsb,alpha,mpiL_extrap,FV)
                x = {'afs': 0}
                priorx = dict()
                for k in fit.p.keys():
                    if k == 'epi':
                        priorx[k] = gv.gvar(0.18220,0.00044)
                    elif k == 'aw0':
                        priorx[k] = gv.gvar(0.7036,0.0005)
                    else:
                        priorx[k] = fit.p[k]
                extrap = fitc.fit_function(x,priorx)
                mean = np.array([j.mean for j in extrap])
                sdev = np.array([j.sdev for j in extrap])
                #print mpiL_extrap
                #print mean
                #print sdev
                mpiL_extrap_plot = np.exp(-mpiL_extrap)/np.sqrt(mpiL_extrap)
                ax.fill_between(mpiL_extrap_plot,mean+sdev,mean-sdev,alpha=0.4,color='#70bf41')
                ax.errorbar(x=mpiL_extrap_plot,y=mean,ls='--',marker='',elinewidth=1,color='#70bf41',label='NLO $\chi$PT prediction')
                return ax, {'mpiL_extrap_plot':mpiL_extrap_plot,'y':extrap}
            def v_data(ax,s,data,result):
                x = data['mpl']
                y = result['fit'].y
                xlist = []
                ylist = []
                elist = []
                for i,e in enumerate(s['ensembles']):
                    if e in ['l2464f211b600m00507m0507m628','l3264f211b600m00507m0507m628','l4064f211b600m00507m0507m628']:
                        xplot = np.exp(-x[i])/np.sqrt(x[i])
                        ax.errorbar(x=xplot,y=y[i].mean,yerr=y[i].sdev,ls='None',marker=self.plot_params[e]['marker'],fillstyle='full',markersize='5',elinewidth=1,capsize=2,color=self.plot_params[e]['color'],label=self.plot_params[e]['label'])
                        xlist.append(xplot)
                        ylist.append(y[i])
                        elist.append(e)
                    else: pass
                return ax, {'x':np.array(xlist),'y':np.array(ylist),'ens':np.array(elist)}
            def v_legend(ax):
                handles, labels = ax.get_legend_handles_labels()
                leg = ax.legend(handles=handles,loc=4,ncol=1, fontsize=12,edgecolor='k',fancybox=False)
                plt.gca().add_artist(leg)
                return None
            r_fv = dict()
            for ansatz_truncate in s['ansatz']['type']:
                if ansatz_truncate.split('_')[0] in ['xpt-delta']:
                    print('CAN NOT PRINT: eps_delta(eps_pi) = unknown')
                    continue
                result = result_list[ansatz_truncate]
                fig = plt.figure('%s infinite volume extrapolation' %ansatz_truncate,figsize=(7,4.326237))
                ax = plt.axes([0.15,0.15,0.8,0.8])
                # plot IV extrapolation
                ax, r0 = v_vol(ax,s,result,ansatz_truncate)
                # plot data
                ax, rd = v_data(ax,s,data,result)
                r_fv[ansatz_truncate] = {'r0':r0,'rd':rd}
                # plot legend
                v_legend(ax)
                # format plot
                ax.set_ylim([1.22,1.3])
                ax.set_xlim([0,0.025])
                ax.set_xlabel('$e^{-m_\pi L}/(m_\pi L)^{1/2}$', fontsize=20)
                ax.set_ylabel('$g_A$', fontsize=20)
                ax.yaxis.set_ticks([1.23,1.25,1.27,1.29])
                ax.xaxis.set_tick_params(labelsize=16)
                ax.yaxis.set_tick_params(labelsize=16)
                ax.set_title(self.title[ansatz_truncate],fontdict={'fontsize':20,'verticalalignment':'top','horizontalalignment':'left'},x=0.05,y=0.9)
                if s['save_figs']:
                    plt.savefig('%s/volume_%s.pdf' %(self.loc,ansatz_truncate),transparent=True)
                plt.draw()
            return r_fv
        else:
            print('no FV prediction')
    def plot_histogram(self,s,pp):
        x = pp['x']
        ysum = pp['pdf']
        ydict = pp['pdfdict']
        cdf = pp['cdf']
        # '-','--','-.',':'
        # #ec5d57 #70bf41 #51a7f9
        p = dict()
        p['taylor_2'] = {'color':'#ec5d57','ls':'--','tag':'NLO Taylor $\epsilon_\pi^2$'}
        p['taylor_4'] = {'color':'#ec5d57','ls':':','tag':'NNLO Taylor $\epsilon_\pi^2$'}
        p['xpt_3']    = {'color':'#70bf41','ls':'--','tag':'NNLO $\chi$PT'}
        p['xpt_4']    = {'color':'#70bf41','ls':':','tag':'N3LO $\chi$PT'}
        p['linear_2'] = {'color':'#51a7f9','ls':'--','tag':'NLO Taylor $\epsilon_\pi$'}
        p['linear_4'] = {'color':'#51a7f9','ls':':','tag':'NNLO Taylor $\epsilon_\pi$'}
        p['xpt-delta_2'] = {'color':'#70bf41','ls':'--','tag':'NLO $\Delta\chi$PT'}
        p['xpt-delta_3'] = {'color':'#70bf41','ls':':','tag':'NNLO $\Delta\chi$PT'}
        p['xpt-delta_4'] = {'color':'#70bf41','ls':'-.','tag':'N3LO $\Delta\chi$PT'}
        fig = plt.figure('result histogram',figsize=(7,4.326237))
        ax = plt.axes([0.15,0.15,0.8,0.8])
        ax.fill_between(x=x,y1=ysum,facecolor='#b36ae2',edgecolor='black',alpha=0.4,label='model average')
        # get 95% confidence
        lidx95 = abs(cdf-0.025).argmin()
        uidx95 = abs(cdf-0.975).argmin()
        ax.fill_between(x=x[lidx95:uidx95],y1=ysum[lidx95:uidx95],facecolor='#b36ae2',edgecolor='black',alpha=0.4)
        # get 68% confidence
        lidx68 = abs(cdf-0.158655254).argmin()
        uidx68 = abs(cdf-0.841344746).argmin()
        ax.fill_between(x=x[lidx68:uidx68],y1=ysum[lidx68:uidx68],facecolor='#b36ae2',edgecolor='black',alpha=0.4)
        # plot black curve over
        ax.errorbar(x=[x[lidx95],x[lidx95]],y=[0,ysum[lidx95]],color='black',lw=2)
        ax.errorbar(x=[x[uidx95],x[uidx95]],y=[0,ysum[uidx95]],color='black',lw=2)
        ax.errorbar(x=[x[lidx68],x[lidx68]],y=[0,ysum[lidx68]],color='black',lw=2)
        ax.errorbar(x=[x[uidx68],x[uidx68]],y=[0,ysum[uidx68]],color='black',lw=2)
        ax.errorbar(x=x,y=ysum,ls='-',color='black')
        for a in ydict.keys():
            ax.errorbar(x=x,y=ydict[a],ls=p[a]['ls'],color=p[a]['color'],label=p[a]['tag'],lw=2)
        ax.legend(fontsize=12,edgecolor='k',fancybox=False)
        ax.set_ylim(bottom=0)
        ax.set_xlim([1.225,1.335])
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(False)
        ax.xaxis.set_tick_params(labelsize=16)
        if s['save_figs']:
            plt.savefig('%s/model_avg_histogram.pdf' %(self.loc),transparent=True)
        plt.draw()
    def model_avg_chiral(self,s,phys,wd,r_chiral):
        # model average
        y = 0
        ya = {0:0,1:0,2:0}
        d = 0
        for k in wd.keys():
            y += wd[k]*r_chiral[k]['r0']['y']
            d += wd[k]*r_chiral[k]['rd']['y']
            for a in r_chiral[k]['ra'].keys():
                ya[a] += wd[k]*r_chiral[k]['ra'][a]
        # plot
        fig = plt.figure('model average chiral extrapolation',figsize=(7,4.326237))
        ax = plt.axes([0.15,0.15,0.8,0.8])
        # physical epi
        F = phys['fpi']/np.sqrt(2)
        m = phys['mpi']
        epi_phys = m/(4.*np.pi*F)
        ax.axvspan(epi_phys.mean-epi_phys.sdev, epi_phys.mean+epi_phys.sdev, alpha=0.4, color='#a6aaa9')
        ax.axvline(epi_phys.mean,ls='--',color='#a6aaa9')
        # continuum extrap
        epi_extrap = r_chiral[k]['r0']['epi']
        mean = np.array([i.mean for i in y])
        sdev = np.array([i.sdev for i in y])
        ax.fill_between(epi_extrap,mean+sdev,mean-sdev,alpha=0.4,color='#b36ae2',label='$g_A^{LQCD}(\epsilon_\pi,a=0)$')
        ax.errorbar(x=epi_extrap,y=mean,ls='--',marker='',elinewidth=1,color='#b36ae2')
        # finite lattice spacing
        pp = self.plot_params
        color_list = [pp['l1648f211b580m013m065m838']['color'], pp['l2464f211b600m0170m0509m635']['color'], pp['l3296f211b630m0074m037m440']['color']]
        label = ['$g_A(\epsilon_\pi,a\simeq 0.15$~fm$)$','$g_A(\epsilon_\pi,a\simeq 0.12$~fm$)$','$g_A(\epsilon_\pi,a\simeq 0.09$~fm$)$']
        for idx,i in enumerate(r_chiral[k]['ra'].keys()):
            ax.errorbar(x=r_chiral[k]['r0']['epi'],y=[j.mean for j in ya[i]],ls='-',marker='',elinewidth=1,color=color_list[idx],label=label[idx])
        # data
        for i,e in enumerate(r_chiral[k]['rd']['ens']):
            ax.errorbar(x=r_chiral[k]['rd']['x'][i].mean,y=d[i].mean,yerr=d[i].sdev,ls='None',marker=self.plot_params[e]['marker'],fillstyle='full',markersize='5',elinewidth=1,capsize=2,color=self.plot_params[e]['color'],label=self.plot_params[e]['label'])
        # pdg
        gA_pdg = [1.2723, 0.0023]
        ax.errorbar(x=epi_phys.mean,y=gA_pdg[0],yerr=gA_pdg[1],ls='None',marker='o',fillstyle='none',markersize='8',capsize=2,color='black',label='$g_A^{PDG}=1.2723(23)$')
        # legend
        handles, labels = ax.get_legend_handles_labels()
        l0 = [handles[0],handles[-1]]
        l1 = [handles[i] for i in range(len(handles)-2,0,-1)]
        leg = ax.legend(handles=l0,numpoints=1,loc=1,ncol=1,fontsize=12,edgecolor='k',fancybox=False)
        ax.legend(handles=l1,numpoints=1,loc=4,ncol=2,fontsize=12,edgecolor='k',fancybox=False)
        plt.gca().add_artist(leg)
        # settings
        ax.set_ylim([1.075,1.375])
        ax.set_xlim([0,0.32])
        ax.set_xlabel('$\epsilon_\pi=m_\pi/(4\pi F_\pi)$', fontsize=20)
        ax.set_ylabel('$g_A$', fontsize=20)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_title('model average',fontdict={'fontsize':20,'verticalalignment':'top','horizontalalignment':'left'},x=0.05,y=0.9)
        if s['save_figs']:
            plt.savefig('%s/chiral_modelavg.pdf' %(self.loc),transparent=True)
        plt.draw()
    def model_avg_cont(self,s,wd,r_cont):
        # model average
        y = 0
        ym = {0:0,1:0,2:0,3:0,4:0}
        d = 0
        for k in wd.keys():
            y += wd[k]*r_cont[k]['r0']['y']
            d += wd[k]*r_cont[k]['rd']['y']
            for a in r_cont[k]['rm'].keys():
                ym[a] += wd[k]*r_cont[k]['rm'][a]
        # plot
        fig = plt.figure('model average chiral extrapolation',figsize=(7,4.326237))
        ax = plt.axes([0.15,0.15,0.8,0.8])
        # physical pion mass extrap
        a_extrap = r_cont[k]['r0']['aw0_extrap_plot']
        mean = np.array([i.mean for i in y])
        sdev = np.array([i.sdev for i in y])
        ax.fill_between(a_extrap,mean+sdev,mean-sdev,alpha=0.4,color='#b36ae2',label='$g_A^{LQCD}(\epsilon_\pi^{phys.},\epsilon_a)$')
        ax.errorbar(x=a_extrap,y=mean,ls='--',marker='',elinewidth=1,color='#b36ae2')
        # unphysical pion masses
        ls_list = ['-','--','-.',':','-']
        label = ['$g_A(\epsilon^{(130)}_\pi,\epsilon_a)$','$g_A(\epsilon^{(220)}_\pi,\epsilon_a)$','$g_A(\epsilon^{(310)}_\pi,\epsilon_a)$','$g_A(\epsilon^{(350)}_\pi,\epsilon_a)$','$g_A(\epsilon^{(400)}_\pi,\epsilon_a)$']
        color = ['black','black','black','black','black']
        dashes = [8, 4, 2, 4, 2, 4]
        for idx,i in enumerate(r_cont[k]['rm'].keys()):
            if i == 4:
                ax.errorbar(x=a_extrap,y=[j.mean for j in ym[i]],ls=ls_list[idx],dashes=dashes,marker='',elinewidth=1,color=color[idx],label=label[idx])
            else:
                ax.errorbar(x=a_extrap,y=[j.mean for j in ym[i]],ls=ls_list[idx],marker='',elinewidth=1,color=color[idx],label=label[idx])
        # data
        for i,e in enumerate(r_cont[k]['rd']['ens']):
            ax.errorbar(x=r_cont[k]['rd']['x'][i].mean,y=d[i].mean,yerr=d[i].sdev,ls='None',marker=self.plot_params[e]['marker'],fillstyle='full',markersize='5',elinewidth=1,capsize=2,color=self.plot_params[e]['color'])
        # pdg
        gA_pdg = [1.2723, 0.0023]
        ax.errorbar(x=0,y=gA_pdg[0],yerr=gA_pdg[1],ls='None',marker='o',fillstyle='none',markersize='8',capsize=2,color='black',label='$g_A^{PDG}=1.2723(23)$')
        # legend
        handles, labels = ax.get_legend_handles_labels()
        l0 = [handles[0],handles[-1]]
        l1 = [handles[i] for i in range(len(handles)-2,0,-1)]
        leg = ax.legend(handles=l0,numpoints=1,loc=1,ncol=1,fontsize=12,edgecolor='k',fancybox=False)
        ax.legend(handles=l1,numpoints=1,loc=3,ncol=2,fontsize=12,edgecolor='k',fancybox=False)
        plt.gca().add_artist(leg)
        # settings
        ax.set_ylim([1.075,1.375])
        ax.set_xlim([-0.001,0.81/(4*np.pi)])
        ax.set_xlabel('$\epsilon_a^2=a^2/(4\pi w^2_0)$', fontsize=20)
        ax.set_ylabel('$g_A$', fontsize=20)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_title('model average',fontdict={'fontsize':20,'verticalalignment':'top','horizontalalignment':'left'},x=0.05,y=0.9)
        if s['save_figs']:
            plt.savefig('%s/cont_modelavg.pdf' %(self.loc),transparent=True)
        plt.draw()
    def model_avg_fv(self,s,wd,r_fv):
        # model average
        y = 0
        ym = {0:0,1:0,2:0,3:0,4:0}
        d = 0
        for k in wd.keys():
            y += wd[k]*r_fv[k]['r0']['y']
            d += wd[k]*r_fv[k]['rd']['y']
        # plot
        fig = plt.figure('model average volume extrapolation',figsize=(7,4.326237))
        ax = plt.axes([0.15,0.15,0.8,0.8])
        # infinite volume extrapolation
        l_extrap = r_fv[k]['r0']['mpiL_extrap_plot']
        mean = np.array([i.mean for i in y])
        sdev = np.array([i.sdev for i in y])
        ax.fill_between(l_extrap,mean+sdev,mean-sdev,alpha=0.4,color='#70bf41')
        ax.errorbar(x=l_extrap,y=mean,ls='--',marker='',elinewidth=1,color='#70bf41',label='NLO $\chi$PT prediction')
        # data
        for i,e in enumerate(r_fv[k]['rd']['ens']):
            ax.errorbar(x=r_fv[k]['rd']['x'][i],y=d[i].mean,yerr=d[i].sdev,ls='None',marker=self.plot_params[e]['marker'],fillstyle='full',markersize='5',elinewidth=1,capsize=2,color=self.plot_params[e]['color'])
        # legend
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles=handles,loc=4,ncol=1, fontsize=12,edgecolor='k',fancybox=False)
        plt.gca().add_artist(leg)
        # settings
        ax.set_ylim([1.22,1.3])
        ax.set_xlim([0,0.025])
        ax.set_xlabel('$e^{-m_\pi L}/(m_\pi L)^{1/2}$', fontsize=20)
        ax.set_ylabel('$g_A$', fontsize=20)
        ax.yaxis.set_ticks([1.23,1.25,1.27,1.29])
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_title('model average',fontdict={'fontsize':20,'verticalalignment':'top','horizontalalignment':'left'},x=0.05,y=0.9)
        if s['save_figs']:
            plt.savefig('%s/fv_modelavg.pdf' %(self.loc),transparent=True)
        plt.draw()
if __name__=='__main__':
    print("chipt library")
