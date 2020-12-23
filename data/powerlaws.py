import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
import time

print('Reading input ... ', end='', flush=True)
t = -time.time()
data = pd.read_csv('day_1M.csv', sep='\t', header=None)
t = t + time.time()
print(f'OK {t:.5} sec')

selector = { 'acc prob': True, 'feat count': True }
cols = range(14,14+26)
#y = data[col].value_counts().to_numpy()
#x = np.arange(1, len(y)+1, 1)
#print(len(y))
#plt.hist(y, bins=200)
##plt.step(x, np.log(y), '.-', where='post', lw=1)
#plt.grid(False)
#plt.show()

if selector['acc prob']:
    for col in cols:
        t = -time.time()
        y = data[col].value_counts().to_list()
        counts = [(key, len(list(val))) for (key, val) in groupby(y)]
        x = [0] * sum(y)
        for i in range(len(counts)):
            key, count = counts[i]
            x[key] = count
        t = t + time.time()
        print(f'Feature {col}\t: {t:.5} sec')
        plt.xlabel('Accesses')
        plt.ylabel('Number of IDs')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.hist(x, log=True, bins=200)
        #plt.show()
        plt.tight_layout()
        plt.savefig(f'fstat_png\\acc_prob_f{col}.png')
        plt.clf()

if selector['feat count']:
    for col in cols:
        t = -time.time()
        x = [int(h,16) for h in data[col].to_list() if h is not np.nan]
        t = t + time.time()
        print(f'Feature {col}\t: {t:.5} sec')
        plt.xticks([])
        plt.xlabel('Embedding IDs')
        plt.ylabel('Accesses')
        plt.hist(x, log=False, bins=200)
        plt.ylim(bottom=0)
        #plt.show()
        plt.tight_layout()
        plt.savefig(f'fstat_png\\feat_count_f{col}.png')
        plt.clf()
