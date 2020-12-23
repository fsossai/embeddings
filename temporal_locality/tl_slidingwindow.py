import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby, compress, tee
import time
import sys
import argparse

parser = argparse.ArgumentParser(
        description='Temporal locality of request of a feature')
parser.add_argument('--feature-index', '-f', type=int, default=14)
parser.add_argument('--window-size', '-w', type=int, default=2000)
parser.add_argument('--top', '-t', type=int, default=3)
args = parser.parse_args()

if args.feature_index not in range(14,39+1):
    print('ERROR: Feature must be in range [14,39]')
    sys.exit(-1)

data = pd.read_csv('..\day_1M.csv', sep='\t', header=None)
wsize = args.window_size

# Missing values are replaced with zeros
column = data[args.feature_index].replace(np.nan, 0)
N = len(column)

# Getting a list of the frequencies of every unique ID
counts = column.value_counts().to_dict()
keys = list(counts)
Z = counts[0] if 0 in counts else 0
print(f'Found {Z} NaN elements ( {100.0 * Z/N:.4}% )')

nan_str = 'NaN'
fig, axs = plt.subplots(1, args.top)

# looping over the first args.top IDs
for id,key in enumerate(keys[0:args.top]):
    column_iter = iter(column)
    print(f'Processing {id+1}-th most frequent ID: {nan_str if key == 0 else key}')
    
    # repeating the index of the window as many times as the occurence of key
    # in that window
    h = [i // wsize for i in range(N - N % wsize)
        if column[i] == key]
    axs[id].hist(h, bins=N//wsize)
    axs[id].set(xlabel='Window index', ylabel='Frequency')
    axs[id].set(yticks=[])
    if key == 0:
        axs[id].set_title(f'{id+1}-th most frequent ID (NaN)')
    else:
        axs[id].set_title(f'{id+1}-th most frequent ID')

for ax in axs.flat:
    ax.label_outer()
fig.suptitle(f'Frequency of top-{args.top} IDs per window, Feature index: {args.feature_index}, Windows size: {wsize}')
plt.tight_layout()
plt.show()