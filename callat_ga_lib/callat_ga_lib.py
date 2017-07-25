import gvar as gv
import scipy.special as spsp
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
                if self.FV:
                    r += self.dfv(p)
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
    return {'result': phys, 'priorc': priorc, 'epi': epi}

def error_budget(s,result):
    fit = result['fit']
    prior = fit.prior
    priorc = result['phys']['priorc']
    phys = result['phys']['result']
    statistical = phys.partialsdev(fit.y)
    input_error = phys.partialsdev(priorc['epi'],prior['aw0'])
    # compile chiral and discretization lists then splat as function input
    if s['ansatz']['FV']:
        X_list = [prior['g0']]
    else:
        X_list = []
    d_list = []
    n = s['ansatz']['truncation']
    for k in prior.keys():
        if (k[0] == 'c' or k[0] == 'b') and int(k[-1]) <= n:
            X_list.append(prior[k])
        if (k[0] == 'a' or k[0] == 's') and k[1] != 'w' and int(k[-1]) <= n:
            d_list.append(prior[k])
    chiral      = phys.partialsdev(*X_list)
    disc        = phys.partialsdev(*d_list)
    err = {'stat': [statistical/phys.mean*100], 'chiral': [chiral/phys.mean*100], 'disc': [disc/phys.mean*100], 'input': [input_error/phys.mean*100], 'total': [phys.sdev/phys.mean*100]}
    return err

class plot_chiral_fit():
    def __init__(self):
        self.plot_params = dict()
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
        self.plot_params['l3296f211b630m0074m037m440']   = {'abbr': 'a09m310',  'color': '#51a7f9', 'marker': 's', 'label': '$a\simeq 0.09$~fm'}
        self.plot_params['l4896f211b630m00363m0363m430'] = {'abbr': 'a09m220',  'color': '#51a7f9', 'marker': '^', 'label': ''}
    def plot_chiral(self,s,data,result):
        # convergence
        def plot_convergence(result,xp):
            fitc = result['fitc']
            x = xp['x']
            priorx = xp['priorx']
            tn = fitc.n
            ls_list = ['-','--','-.',':']
            label = ['LO','NLO','NNLO','N3LO']
            phys_converge = []
            for n in range(1,tn+1):
                fitc.n = n
                extrap = fitc.fit_function(x,priorx)
                # print numerical breakdown
                converge_prior = dict(priorx)
                converge_prior['epi'] = result['phys']['epi']
                phys_converge.append(fitc.fit_function(x,converge_prior))
                if n == 1:
                    extrap = [extrap for i in range(len(priorx['epi']))]
                mean = [i.mean for i in extrap]
                ax.errorbar(x=priorx['epi'],y=mean,ls=ls_list[n-1],marker='',color='black',label=label[n-1])
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
            return ax
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
            return ax, {'x':x, 'priorx':priorx}
        def c_data(ax,s,result):
            x = result['fit'].prior['epi']
            if s['ansatz']['FV']:
                y = result['fit'].y - result['fitc'].dfv(result['fit'].p)
            else:
                y = result['fit'].y
            for i,e in enumerate(s['ensembles']):
                ax.errorbar(x=x[i].mean,xerr=x[i].sdev,y=y[i].mean,yerr=y[i].sdev,ls='None',marker=self.plot_params[e]['marker'],fillstyle='full',markersize='5',elinewidth=1,capsize=2,color=self.plot_params[e]['color'],label=self.plot_params[e]['label'])
            return ax
        def c_pdg(ax,result):
            gA_pdg = [1.2723, 0.0023]
            ax.errorbar(x=result['phys']['epi'].mean,y=gA_pdg[0],yerr=gA_pdg[1],ls='None',marker='o',fillstyle='none',markersize='8',capsize=2,color='black',label='$g_A^{PDG}=1.2723(23)$')
            return ax
        def c_legend(ax):
            handles, labels = ax.get_legend_handles_labels()
            l0 = [handles[0],handles[-1]]
            l1 = [handles[i] for i in range(len(handles)-2,0,-1)]
            leg = ax.legend(handles=l0,numpoints=1,loc=1,ncol=1)
            ax.legend(handles=l1,numpoints=1,loc=4,ncol=2)
            plt.gca().add_artist(leg)
            return None
        ### Chiral extrapolation
        fig = plt.figure('chiral extrapolation',figsize=(7,4.326237))
        ax = plt.axes([0.15,0.15,0.8,0.8])
        # continuum extrapolation
        ax, xp = c_continuum(ax,result) # xp is used to make chipt convergence plot
        # plot chiral extrapolation
        ax = c_chiral(ax,result)
        # plot data
        ax = c_data(ax,s,result)
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
        plt.draw()
        ### Convergence
        fig = plt.figure('chiral convergence',figsize=(7,4.326237))
        ax = plt.axes([0.15,0.15,0.8,0.8])
        ax = plot_convergence(result,xp)
        # plot physical pion point
        epi_phys = result['phys']['epi']
        ax.axvspan(epi_phys.mean-epi_phys.sdev, epi_phys.mean+epi_phys.sdev, alpha=0.4, color='#a6aaa9')
        ax.axvline(epi_phys.mean,ls='--',color='#a6aaa9')
        # make legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles,loc=3,ncol=2)
        # format plot
        ax.set_ylim([1.075,1.375])
        ax.set_xlim([0,0.32])
        ax.set_xlabel('$\epsilon_\pi=m_\pi/(4\pi F_\pi)$', fontsize=20)
        ax.set_ylabel('$g_A$', fontsize=20)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.draw()
    def plot_continuum(self,s,data,result):
        fig = plt.figure('continuum extrapolation',figsize=(7,4.326237))
        ax = plt.axes([0.15,0.15,0.8,0.8])
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
            return ax
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
            return ax, {'x':x, 'priorx':priorx}
        def a_data(ax,s,result):
            x = result['fit'].prior['aw0']
            if s['ansatz']['FV']:
                y = result['fit'].y - result['fitc'].dfv(result['fit'].p)
            else:
                y = result['fit'].y
            for i,e in enumerate(s['ensembles']):
                xplot = x[i]**2/(4.*np.pi)
                ax.errorbar(x=xplot.mean,xerr=xplot.sdev,y=y[i].mean,yerr=y[i].sdev,ls='None',marker=self.plot_params[e]['marker'],fillstyle='full',markersize='5',elinewidth=1,capsize=2,color=self.plot_params[e]['color']) 
            return ax
        def a_pdg(ax,result):
            gA_pdg = [1.2723, 0.0023]
            ax.errorbar(x=0,y=gA_pdg[0],yerr=gA_pdg[1],ls='None',marker='o',fillstyle='none',markersize='8',capsize=2,color='black',label='$g_A^{PDG}=1.2723(23)$')
            return ax
        def a_legend(ax):
            handles, labels = ax.get_legend_handles_labels()
            l0 = [handles[0],handles[-1]]
            l1 = [handles[i] for i in range(len(handles)-2,0,-1)]
            leg = ax.legend(handles=l0,numpoints=1,loc=1,ncol=1)
            ax.legend(handles=l1,numpoints=1,loc=3,ncol=2)
            plt.gca().add_artist(leg)
            return None
        # continuum extrapolation
        ax, res = a_cont(ax,result)
        # chiral extrapolation
        ax = a_chiral(ax,result)
        # plot data
        ax = a_data(ax,s,result)
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
        plt.draw()
    def plot_volume(self,s,data,result):
        if s['ansatz']['FV']:
            fig = plt.figure('infinite volume extrapolation',figsize=(7,4.326237))
            ax = plt.axes([0.15,0.15,0.8,0.8])
            def v_vol(ax,s,result):
                fit = result['fit']
                ansatz = s['ansatz']['type']
                truncate = s['ansatz']['truncation']
                xsb = s['ansatz']['xsb']
                alpha = s['ansatz']['alpha']
                FV = s['ansatz']['FV']
                mpiL_extrap = np.linspace(0.000001,10.200001,10201)
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
                mpiL_extrap_plot = np.exp(-mpiL_extrap)/np.sqrt(mpiL_extrap)
                ax.fill_between(mpiL_extrap_plot,mean+sdev,mean-sdev,alpha=0.4,color='#70bf41')
                ax.errorbar(x=mpiL_extrap_plot,y=mean,ls='--',marker='',elinewidth=1,color='#70bf41',label='NLO $\chi$PT prediction')
                return ax
            def v_data(ax,s,data,result):
                x = data['mpl']
                y = result['fit'].y 
                for i,e in enumerate(s['ensembles']):
                    if e in ['l2464f211b600m00507m0507m628','l3264f211b600m00507m0507m628','l4064f211b600m00507m0507m628']:
                        xplot = np.exp(-x[i])/np.sqrt(x[i])
                        ax.errorbar(x=xplot,y=y[i].mean,yerr=y[i].sdev,ls='None',marker=self.plot_params[e]['marker'],fillstyle='full',markersize='5',elinewidth=1,capsize=2,color=self.plot_params[e]['color'],label=self.plot_params[e]['label'])
                    else: pass
                return ax
            def v_legend(ax):
                handles, labels = ax.get_legend_handles_labels()
                leg = ax.legend(handles=handles,loc=2,ncol=1, fontsize=20,edgecolor='k',fancybox=False)
                plt.gca().add_artist(leg)
                return None
            # plot IV extrapolation
            ax = v_vol(ax,s,result)
            # plot data
            ax = v_data(ax,s,data,result)
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
            plt.draw()
        else:
            print('no FV prediction')

if __name__=='__main__':
    print("chipt library")
