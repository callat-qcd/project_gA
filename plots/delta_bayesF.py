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

f = open('../data/delta_fits.csv','r')
next(f)
fig = plt.figure('delta Bayes Factors',figsize=fig_size2)
ax = plt.axes(plt_axes)
lgbf_max = 0
lgbf_list = []
x = []
for l in f:
    dat = l.split(',')
    x.append(int(dat[0]))
    logGBF = float(dat[1])
    lgbf_list.append(logGBF)
    if logGBF > lgbf_max:
        lgbf_max = logGBF
lgbf_list = np.array(lgbf_list)
x = np.array(x)
bf = np.exp(np.array(lgbf_list)-np.array(lgbf_max))
ax.plot([i for idx,i in enumerate(x) if idx not in [np.argmax(bf)]],\
    [i for idx,i in enumerate(bf) if idx not in [np.argmax(bf)]],\
    marker='s',fillstyle='none',markersize=ms,lw=lw,color='#51a7f9')
ax.plot(x[np.argmax(bf)],bf[np.argmax(bf)],marker='*',color='#c82506',\
    fillstyle='full',markersize=ms_big)

ax.set_ylabel(r'Bayes factor', fontsize=fs_xy)
ax.set_xlabel(r'prior width (\% of central value)', fontsize=fs_xy)
ax.xaxis.set_tick_params(labelsize=ts,width=lw)
ax.yaxis.set_tick_params(labelsize=ts,width=lw)
[i.set_linewidth(lw) for i in ax.spines.itervalues()]

plt.draw()
plt.savefig('./delta_BF.pdf',transparent=True)
plt.show()
