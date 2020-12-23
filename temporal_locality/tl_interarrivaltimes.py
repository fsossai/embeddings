import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby, compress, tee
from scipy import stats 
import time
import sys
import argparse

parser = argparse.ArgumentParser(
        description='Histograms of interarrival times for an index of a sparse feature'
    )
parser.add_argument('--feature-index', '-f', type=int, default=14)
parser.add_argument('--top', '-t', type=int, default=1)
args = parser.parse_args()

if args.feature_index not in range(14,39+1):
    print('ERROR: Feature must be in range [14,39]')
    sys.exit(-1)

data = pd.read_csv('..\day_1M.csv', sep='\t', header=None)

# selecting the (args.top-1)-most frequent ID
key = list(data[args.feature_index]
        .replace(np.nan,0)
        .value_counts()
        .to_dict())[args.top - 1]
print(f'key: {key}')
occurrences = np.where(data[args.feature_index] == key)
print(f'#occurrences: {len(occurrences[0])}')
print(f'Poisson 1/lambda guess\t: {(len(data)-0)/ len(occurrences[0]):.15}')
# calculating interarrival times
a,b = tee(occurrences[0])
next(b, None)
inter = list(map(lambda t1,t2: t2-t1, a, b))
# creating pairs (interarrival time, how many)
inter.sort()
xy = list(zip(*[(key,len(list(val))) for key,val in groupby(inter)]))
x = np.array(xy[0])
y = np.array(xy[1])

# fitting data with an exponential distribution with SciPy
mean = np.mean(inter)
loc, scale = stats.expon.fit(inter)
lam = 1 / scale
max_y = np.max(y) / len(inter) * 1.05
#lam = 1 / mean
print(f'Emprirical mean\t\t: {mean:.15}')
print(f'SciPy fit scale\t\t: {scale:.15}')
print(f'SciPy fit loc\t\t: {loc:.15}')
print(f'mean - (scale + loc)\t: {mean-(scale+loc)}')

plt.bar(x, y / len(inter), color=(0.2, 0.4, 0.8, 0.5),
    label=f'{args.top}-th most frequent embedding ID', width=1)

#plt.plot(x, lam * np.exp(-lam * (x-loc)), 'r-', label='MLE exponential fit')
plt.plot(x, (1/mean) * np.exp(-x / mean), 'r-', label='MLE exponential fit')
plt.plot([mean, mean], [0, max_y], 'b-.', lw=1.3, label='Empirical mean')
#plt.xlim(left=0.5, right=250)
plt.title(f'Interarrival times distribution, feature {args.feature_index}')
plt.xlabel('Interarrival time (discrete units)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()


