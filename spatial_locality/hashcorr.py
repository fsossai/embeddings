import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations, tee
from time import time
import hashlib
#from multiprocessing import Process

input = '..\data\day_100k.csv'
data = pd.read_csv(input, sep='\t', header=None)
data = data.replace(np.nan, '0')
N = len(data)
# for each pair of sparse feature indexes
hcs = []
print('Counting unique values ... ', end='', flush=True)
t = -time()
counts = [data[i].value_counts().size for i in range(14,14+26)]
t += time()
print(f'{t:.5} sec')

# pairs of features to be processed
pairs = combinations(range(14, 14+26), r=2) # every pair
#all = list(zip(*sorted(zip(counts,range(14, 14+26)), reverse=True)))[1]
#f1,f2 = iter(all), iter(all)
#next(f2)
#pairs = zip(f1,f2)

feat_start = 14
t = -time()
for i,j in pairs:
    print(f'\rProcessing feature {i} with {j} ... ', end='')
    L = data[[i,j]].value_counts().size
    m = min(counts[i-feat_start] * counts[j-feat_start], N)
    M = max(counts[i-feat_start], counts[j-feat_start])
    hc = (m-L) / (m-M)
    hcs.append( ((i,j), hc) )
t += time()
print('\r',*([' ']*60), end='') # clearing 60 chars of the last line
print(f'\rDone in {t:.8} sec')

# print top-k
hcs.sort(key=lambda x: x[1], reverse=True)
k = 10
print(f'Top-{k} correlations')
for pair,hc in hcs[0:k]:
    print(f'{pair}\t: {100*hc:.5}%')

# saving to file
with open('hc.txt','a') as f:
    f.write(f'Hash correlations in \'{input}\'\n')
    for pair,hc in hcs:
        f.write(f'{pair}\t: {100*hc:.5}%\n')
    f.write('\n\n')


