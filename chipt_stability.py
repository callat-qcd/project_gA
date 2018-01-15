import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

f = open('./data/stability.csv','r')
next(f)
#fig = plt.figure('chiral extrapolation',figsize=(7,4.326237))
fig = plt.figure('chiral extrapolation',figsize=(7,6))
ax0 = plt.axes([0.25,0.1,0.4,0.85])
ax1 = plt.axes([0.66,0.1,0.14,0.85])
ax2 = plt.axes([0.81,0.1,0.14,0.85])
pdat = next(f).split(',')
ax0.axvspan(float(pdat[2])-float(pdat[3]),float(pdat[2])+float(pdat[3]),alpha=0.4,color='#b36ae2')
ticklabels = []
ticks = []
chisq_max = 0
lgbf_max = 0
y = 0
lgbf_list = []
avg_list = []
label_list = []
for l in f:
    if l.split()[0] == 'break':
        y -= 1
    else:
        y -= 1
        dat = l.split(',')
        ticks.append(y)
        if dat[0] == 'True':
            avg = True
        else:
            avg = False
        label = dat[1]
        avg_list.append(avg)
        label_list.append(label)
        mean = float(dat[2])
        sdev = float(dat[3])
        chi2dof = float(dat[4])
        if chi2dof > chisq_max:
            chisq_max = chi2dof
        logGBF = float(dat[5])
        lgbf_list.append(logGBF)
        if logGBF > lgbf_max and logGBF not in [10000]:
            lgbf_max = logGBF
        lbl = label
        mrk = 'o'
        clr = 'k'
        alpha=0.5
        if label in ['N3LO $\chi$PT','NLO $\chi$PT($\Delta$)']:
            alpha = 1
            clr = '#b36ae2'
        if avg:
            mfc = '#b36ae2'
            clr = '#b36ae2'
            mrk = 's'
            alpha=1
        else:
            mfc = 'None'
        if label=='weighted avg':
            mrk = 's'
            clr = 'k'
            mfc = 'k'
            lbl = r"\textbf{model avg}"
            alpha=1

        ax0.errorbar(x=mean,xerr=sdev,y=y,ls='None',marker=mrk,mfc=mfc,markersize='5',\
            elinewidth=1,capsize=2,color=clr,mew=1,alpha=alpha)
        ax1.errorbar(x=chi2dof,y=y,ls='None',marker=mrk,mfc=mfc,markersize='5',\
            elinewidth=1,capsize=2,color=clr,mew=1,alpha=alpha)
        #ax2.errorbar(x=logGBF,y=y,ls='None',marker=mrk,mfc=mfc,markersize='5',elinewidth=1,capsize=2,color=clr,mew=1)
        ticklabels.append(lbl)
bf = np.exp(np.array(lgbf_list)-np.array(lgbf_max))
print(bf)
for idx,t in enumerate(ticks):
    if avg_list[idx]:
        mfc = '#b36ae2'
        #lbl = r'\textbf{%s}' %label
        mrk = 's'
    else:
        mfc = 'None'
    if label_list[idx]=='weighted avg':
        mrk = 's'
        clr = 'k'
        mfc = 'k'
        lbl = r"\textbf{model avg}"
    else:
        clr = '#b36ae2'
    ax2.errorbar(x=bf[idx],y=t,ls='None',marker=mrk,mfc=mfc,markersize='5',elinewidth=1,capsize=2,color=clr,mew=1)

#print ticklabels
ax0.set_xlabel('$g_A$', fontsize=16)
ax1.set_xlabel('$\chi^2_{\mathrm{aug}}/$dof', fontsize=16)
ax2.set_xlabel('Bayes factor', fontsize=16)
ax0.set_yticks(ticks)
ax0.set_yticklabels(ticklabels)
ax1.set_yticks(ticks)
ax2.set_yticks(ticks)
ax0.set_xticks([1.24,1.28,1.32,1.36])
ax1.set_xticks([0.4,1,1.6])
ax2.set_xticks([0,0.5,1])
ax0.set_ylim([ticks[-1]-1,ticks[0]+1])
ax0.set_xlim([1.23,1.37])
ax1.set_ylim([ticks[-1]-1,ticks[0]+1])
ax1.set_xlim([0,chisq_max*1.1])
ax2.set_ylim([ticks[-1]-1,ticks[0]+1])
ax2.set_xlim([-0.1,1.1])
ax0.xaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(labelsize=12,direction='in')
ax1.xaxis.set_tick_params(labelsize=12,direction='in')
ax2.xaxis.set_tick_params(labelsize=12,direction='in')
ax0.yaxis.set_tick_params(labelsize=12,direction='in')
ax1.yaxis.set_tick_params(labelleft=False,direction='in')
ax2.yaxis.set_tick_params(labelleft=False,direction='in')
plt.draw()
plt.savefig('./plots/chipt_stability.pdf',transparent=True)
plt.show()
