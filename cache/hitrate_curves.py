"""
Plotting the results obtained with a stack simulation.
The files to be plotted are in CSV format, one file
for each categorical feature.
"""

import matplotlib.pyplot as plt
import pandas as pd
import argparse
from glob import glob
import re
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-F', type=str, required=True)
    args = parser.parse_args()

    all_files = glob(args.files)
    if len(all_files) == 0:
        print('ERROR: files not found')
        sys.exit(-1)
  
    # importing into DataFrame all CSV files of the hitrates
    hitrates = dict()
    for file in all_files:
        index = int(re.findall(r'\d+', file)[-1])
        hitrates[index] = pd.read_csv(file)
   
    feats = [33, 14, 35, 23, 34, 24, 36, 25]
    N1, N2 = 2, 4
    fig, axs = plt.subplots(N1, N2)

    # The four most frequent
    k = 0
    for i in range(N1):
        for j in range(N2):
            axs[i, j].plot(hitrates[feats[k]]['size'],
                hitrates[feats[k]]['hitrate'])
            axs[i, j].set(ylim=[0, 1], title='Feature ' + str(feats[k]))
            if i == N1-1:
                axs[i, j].set(xlabel='Cache size (elements)')
            if j == 0:
                axs[i, j].set(ylabel='Hit-rate')
            k += 1

    fig.suptitle('LRU Hit-rate curves')
    plt.tight_layout()
    plt.show()
