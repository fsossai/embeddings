"""
Plotting the results obtained with a simulation of
the LRU and LFU policies.
The files to be plotted are in CSV format, one file
for each categorical feature.
"""

import matplotlib.pyplot as plt
import pandas as pd
import argparse
from glob import glob
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-f', type=str, required=True)
    args = parser.parse_args()

    all_files = glob(args.files)

    # importing into DataFrame all CSV files of the hitrates
    hitrates = dict()
    for file in all_files:
        index = int(re.findall(r'\d+', file)[-1])
        hitrates[index] = pd.read_csv(file)

    feats = [33, 14, 35, 23] + [34, 24, 36, 25]
    #feats = [19, 30, 26, 32] + [39, 22, 38, 29]
    N1, N2 = 2, 4
    fig, axs = plt.subplots(N1, N2)

    # plotting
    k = 0
    for i in range(N1):
        for j in range(N2):
            axs[i, j].plot(hitrates[feats[k]]['cache_size_relative']*100,
                hitrates[feats[k]]['hitrate_LRU'], '.-', label='LRU', )
            axs[i, j].plot(hitrates[feats[k]]['cache_size_relative']*100,
                hitrates[feats[k]]['hitrate_LFU'], '.-', label='LFU', )
            axs[i, j].plot(hitrates[feats[k]]['cache_size_relative']*100,
                hitrates[feats[k]]['hitrate_OPT'], '.-', label='OPT', )
            k += 1
    
    # setting axis limits
    ylim = [0, 1]
    xticks = range(0,26,5)
    k = 0
    for i in range(N1):
        for j in range(N2):
            axs[i, j].set(ylim=ylim, xticks=xticks,
                title='Feature ' + str(feats[k]))
            k += 1

    plt.legend()
    fig.suptitle('LRU/LFU/OPT Hit-rate curves comparison')
    plt.tight_layout()
    plt.show()
