import numpy as np
import matplotlib.pyplot as plt

f = open('./data/stability.csv','r')
next(f)
fig = plt.figure('chiral extrapolation',figsize=(7,4.326237))
#fig = plt.figure('chiral extrapolation',figsize=(7,5))
ax0 = plt.axes([0.25,0.15,0.4,0.8])
ax1 = plt.axes([0.66,0.15,0.14,0.8])
ax2 = plt.axes([0.81,0.15,0.14,0.8])
pdat = next(f).split(',')
ax0.axvspan(float(pdat[2])-float(pdat[3]),float(pdat[2])+float(pdat[3]),alpha=0.4,color='#b36ae2')
ticklabels = []
ticks = []
chisq_max = 0
lgbf_max = 0
y = 0
for l in f:
    if l.split()[0] == 'break':
        y -= 1
    else:
        y -= 1
        dat = l.split(',')
        print dat
        ticks.append(y)
        if dat[0] == 'True':
            avg = True
        else:
            avg = False
        label = dat[1]
        #print label
        mean = float(dat[2])
        sdev = float(dat[3])
        chi2dof = float(dat[4])
        if chi2dof > chisq_max:
            chisq_max = chi2dof
        logGBF = float(dat[5])
        if logGBF > lgbf_max:
            lgbf_max = logGBF
        #if label=='NNNLO $\chi$PT':
        #    ax0.errorbar(x=mean,xerr=sdev,y=y,ls='None',marker='s',fillstyle='full',markersize='5',elinewidth=1,capsize=2,color='black',mew=1)
        #    ax1.errorbar(x=chi2dof,y=y,ls='None',marker='s',fillstyle='full',markersize='5',elinewidth=1,capsize=2,color='black',mew=1)
        #    ax1.axvline(x=chi2dof,ls='--',color='black')
        #    ax2.errorbar(x=logGBF,y=y,ls='None',marker='s',fillstyle='full',markersize='5',elinewidth=1,capsize=2,color='black',mew=1)
        #    ax2.axvline(x=logGBF,ls='--',color='black')
        lbl = label
        mrk = 'o'
        if avg:
            mfc = '#b36ae2'
            #lbl = r'\textbf{%s}' %label
            mrk = 's'
        else:
            mfc = 'None'
        if label=='weighted avg':
            mrk = 's'
            clr = 'k'
            mfc = 'k'
            lbl = r'\textbf{weighted avg}'
        else:
            clr = '#b36ae2'
        ax0.errorbar(x=mean,xerr=sdev,y=y,ls='None',marker=mrk,mfc=mfc,markersize='5',elinewidth=1,capsize=2,color=clr,mew=1)
        ax1.errorbar(x=chi2dof,y=y,ls='None',marker=mrk,mfc=mfc,markersize='5',elinewidth=1,capsize=2,color=clr,mew=1)
        ax2.errorbar(x=logGBF,y=y,ls='None',marker=mrk,mfc=mfc,markersize='5',elinewidth=1,capsize=2,color=clr,mew=1)
        ticklabels.append(lbl)
#print ticklabels
ax0.set_xlabel('$g_A$', fontsize=20)
ax1.set_xlabel('$\chi^2/$dof', fontsize=20)
ax2.set_xlabel('logGBF', fontsize=20)
ax0.set_yticks(ticks)
ax0.set_yticklabels(ticklabels)
ax1.set_yticks(ticks)
ax2.set_yticks(ticks)
ax0.set_xticks([1.2,1.24,1.28,1.32,1.36])
ax1.set_xticks([0.5,1])
ax2.set_xticks([0,10,20])
ax0.set_ylim([ticks[-1]-1,ticks[0]+1])
ax0.set_xlim([1.2,1.35])
ax1.set_ylim([ticks[-1]-1,ticks[0]+1])
ax1.set_xlim([0,chisq_max*1.1])
ax2.set_ylim([ticks[-1]-1,ticks[0]+1])
ax2.set_xlim([0,lgbf_max*1.1])
ax0.xaxis.set_tick_params(labelsize=16)
ax1.xaxis.set_tick_params(labelsize=16)
ax2.xaxis.set_tick_params(labelsize=16)
ax0.yaxis.set_tick_params(labelsize=10)
ax1.yaxis.set_tick_params(labelleft=False)
ax2.yaxis.set_tick_params(labelleft=False)
plt.draw()
plt.savefig('chipt_stability.pdf',transparent=True)
plt.show()
