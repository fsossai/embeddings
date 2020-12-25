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
parser.add_argument('--id', '-i', type=str, default=None)
args = parser.parse_args()

single = args.id is not None

if args.feature_index not in range(14,39+1):
    print('ERROR: Feature must be in range [14,39]')
    sys.exit(-1)

data = pd.read_csv('..\data\day_100k.csv', sep='\t', header=None)
wsize = args.window_size

column = data[args.feature_index].replace(np.nan, 0)
N = len(column)

# Getting a list of the frequencies of every unique ID
vcounts = column.value_counts()
keys = [k for k,_ in vcounts.iteritems()]

nan_str = 'NaN'
if single:
    fig, axs = plt.subplots(1, 1)
    axs = np.array([axs])
else:
    fig, axs = plt.subplots(1, args.top)

if single:
    # lookup for the specific id
    index = next((
        i for i,k in enumerate(keys)
        if k == args.id
    ), None)
    if index is None:
        raise Exception(f'ID {args.id} not found')
    print(f'\'{args.id}\' found as the {index+1}-th most frequent, ' + 
        f'appearing {vcounts[args.id]} times')
    # sliding window
    h = [i // wsize for i in range(N - N % wsize)
            if column[i] == args.id]
    axs[0].hist(h, bins=N//wsize)
    axs[0].set(xlabel='Window index', ylabel='Frequency')
    axs[0].set(yticks=[])
else:
    # looping over the first args.top IDs
    for id,key in enumerate(keys[0:args.top]):
        print(f'Processing {id+1}-th most frequent ID: {key}')

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