import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

f = open('./data/delta_fits.csv','r')
next(f)
fig = plt.figure('delta Bayes Factors',figsize=(7,4.326237))
ax0 = plt.axes([0.1,0.15,0.85,0.8])
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
ax0.plot(x,bf,marker='o')
ax0.plot(x[np.argmax(bf)],bf[np.argmax(bf)],marker='s',color='k')

ax0.set_ylabel(r'Bayes Factor', fontsize=16)
ax0.set_xlabel(r'\% prior width', fontsize=16)
plt.draw()
plt.savefig('./plots/delta_BF.pdf',transparent=True)
plt.show()
