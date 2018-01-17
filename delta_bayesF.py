import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
if True:
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{helvet}',
        r'\usepackage{sansmath}',
        r'\sansmath']

f = open('./data/delta_fits.csv','r')
next(f)
fig = plt.figure('delta Bayes Factors',figsize=(7,4.326237))
ax0 = plt.axes([0.15,0.15,0.8,0.8])
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
ax0.plot([i for idx,i in enumerate(x) if idx not in [np.argmax(bf)]],[i for idx,i in enumerate(bf) if idx not in [np.argmax(bf)]],marker='s',fillstyle='none',markersize='5',color='#51a7f9')
ax0.plot(x[np.argmax(bf)],bf[np.argmax(bf)],marker='*',color='#c82506',fillstyle='full',markersize='8')

ax0.set_ylabel(r'Bayes factor', fontsize=20)
ax0.set_xlabel(r'prior width (\% of central value)', fontsize=20)
ax0.xaxis.set_tick_params(labelsize=16)
ax0.yaxis.set_tick_params(labelsize=16)
plt.draw()
plt.savefig('./plots/delta_BF.pdf',transparent=True)
plt.show()
