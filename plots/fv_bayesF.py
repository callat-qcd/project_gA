import os
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
nature_figs=True
if nature_figs:
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{helvet}',
        r'\usepackage{sansmath}',
        r'\sansmath']
    ms = '3'
    ms_big = '6'
    cs = 3
    fs_l = 7
    fs_xy = 7
    ts = 7
    lw = 0.5
    gr = 1.618034333
    fs2_base = 3.50394
    fig_size2 = (fs2_base,fs2_base/gr)
    fs3_base = 3.50394 #2.40157
    fig_size3 = (fs3_base,fs3_base/gr)
else:
    ms = '5'
    ms_big = '8'
    cs = 2
    fs_l = 12
    fs_xy = 20
    ts = 16
    lw = 1
    gr = 1.618034333
    fs2_base = 7
    fig_size2 = (fs2_base,fs2_base/gr)
    fs3_base = 7 #4.66666667
    fig_size3 = (fs3_base,fs3_base/gr)
plt_axes = [0.14,0.165,0.825,0.825]
if not os.path.exists('plots'):
    os.makedirs('plots')

f = open('../data/fv_fits.csv','r')
header = np.array(f.readline().split(','))[1:]
header[-1] = header[-1].strip('\n')
tag = np.array(f.readline().split(','))[1:]
tag[-1] = tag[-1].strip('\n')
lgbf_max = {k:0 for k in header}
lgbf_list = {k:[] for k in header}
x = []
for l in f:
    dat = np.array(l.split(','))
    x.append(float(dat[0]))
    logGBF = np.array([float(i) for i in dat[1:]])
    for idx,k in enumerate(header):
        lgbf_list[k].append(logGBF[idx])
        if logGBF[idx] > lgbf_max[k]:
            lgbf_max[k] = logGBF[idx]
lgbf_list = {k:np.array(lgbf_list[k]) for k in lgbf_list}
print('Max Bayes Factor for each fit')
for k in lgbf_list:
    print(k,lgbf_max[k])
x = np.array(x)
bf = {k:np.exp(np.array(lgbf_list[k])-np.array(lgbf_max[k])) for k in lgbf_list}

# plot
title_y_pos = {'taylor_2':['top',0.9],'taylor_4':['top',0.9],\
    'linear_2':['top',0.9],'linear_4':['top',0.9],\
    'xpt_3':['bottom',0.1],'xpt_4':['bottom',0.1]}
tag = {'taylor_2':r'NLO Taylor $\epsilon_\pi^2$','taylor_4':r'NNLO Taylor $\epsilon_\pi^2$',
    'linear_2':r'NLO Taylor $\epsilon_\pi$','linear_4':r'NNLO Taylor $\epsilon_\pi$',
    'xpt_3':r'NNLO $\chi$PT','xpt_4':r'NNLO+ct $\chi$PT'}
for ik,k in enumerate(lgbf_list):
    fig = plt.figure('%s finite volume Bayes Factors' %k,figsize=fig_size2)
    ax = plt.axes(plt_axes)
    ax.plot([i for idx,i in enumerate(x) if idx not in [np.argmax(bf[k])]],\
        [i for idx,i in enumerate(bf[k]) if idx not in [np.argmax(bf[k])]],\
        marker='s',fillstyle='none',markersize=ms,color='#51a7f9')
    ax.plot(x[np.argmax(bf[k])],bf[k][np.argmax(bf[k])],marker='*',color='#c82506',fillstyle='full',markersize=ms_big)
    print(x[np.argmax(bf[k])])
    ax.set_ylabel(r'Bayes factor', fontsize=fs_xy)
    ax.set_xlabel(r'NNLO finite volume prior width', fontsize=fs_xy)
    ax.xaxis.set_tick_params(labelsize=ts,width=lw)
    ax.yaxis.set_tick_params(labelsize=ts,width=lw)
    ax.set_title('%s' %tag[k],fontdict={'fontsize':fs_xy,\
        'verticalalignment':title_y_pos[k][0],'horizontalalignment':'right'},\
        x=0.95,y=title_y_pos[k][1],bbox=dict())
    #plt.draw()
    [i.set_linewidth(lw) for i in ax.spines.itervalues()]
    plt.savefig('./%s_fv_BF.pdf' %k,transparent=True)
plt.show()
