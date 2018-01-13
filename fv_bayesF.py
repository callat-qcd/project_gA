import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

f = open('./data/fv_fits.csv','r')
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
x = np.array(x)
bf = {k:np.exp(np.array(lgbf_list[k])-np.array(lgbf_max[k])) for k in lgbf_list}

# plot 
title_y_pos = [['bottom',0.1],['bottom',0.1],['top',0.9],['top',0.9],['top',0.9],['top',0.9]]
for idx,k in enumerate(lgbf_list):
    fig = plt.figure('%s finite volume Bayes Factors' %k,figsize=(7,4.326237))
    ax0 = plt.axes([0.15,0.15,0.8,0.8])
    ax0.plot([i for idx,i in enumerate(x) if idx not in [np.argmax(bf[k])]],[i for idx,i in enumerate(bf[k]) if idx not in [np.argmax(bf[k])]],marker='s',fillstyle='none',markersize='5',color='#51a7f9')
    ax0.plot(x[np.argmax(bf[k])],bf[k][np.argmax(bf[k])],marker='*',color='#c82506',fillstyle='full',markersize='8')
    print(x[np.argmax(bf[k])])
    ax0.set_ylabel(r'Bayes factor', fontsize=20)
    ax0.set_xlabel(r'NNLO finite volume prior width', fontsize=20)
    ax0.xaxis.set_tick_params(labelsize=16)
    ax0.yaxis.set_tick_params(labelsize=16)
    ax0.set_title('%s' %tag[idx],fontdict={'fontsize':20,'verticalalignment':title_y_pos[idx][0],'horizontalalignment':'right'},x=0.95,y=title_y_pos[idx][1],bbox=dict())
    #plt.draw()
    plt.savefig('./plots/%s_fv_BF.pdf' %k,transparent=True)
    plt.show()
