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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-F', type=str, required=True)
    args = parser.parse_args()

    all_files = glob(args.files)

    # importing into DataFrame all CSV files of the hitrates
    hitrates = dict()
    for file in all_files:
        index = int(re.findall(r'\d+', file)[-1])
        hitrates[index] = pd.read_csv(file)

    feats = [33, 14, 35, 23]
    # feats = [34, 24, 36, 25]
    fig, axs = plt.subplots(2, 2)

    # The four most frequent
    axs[0, 0].plot(hitrates[feats[0]]['size'], hitrates[feats[0]]['hitrate'])
    axs[0, 1].plot(hitrates[feats[1]]['size'], hitrates[feats[1]]['hitrate'])
    axs[1, 0].plot(hitrates[feats[2]]['size'], hitrates[feats[2]]['hitrate'])
    axs[1, 1].plot(hitrates[feats[3]]['size'], hitrates[feats[3]]['hitrate'])

    axs[0, 0].set(ylim=[0, 1])
    axs[0, 1].set(ylim=[0, 1])
    axs[1, 0].set(ylim=[0, 1])
    axs[1, 1].set(ylim=[0, 1])

    axs[0, 0].set(ylabel='Hit-rate', xscale='log')
    axs[0, 1].set(xscale='log')
    axs[1, 0].set(xlabel='Cache size (elements)', ylabel='Hit-rate', xscale='log')
    axs[1, 1].set(xlabel='Cache size (elements)', xscale='log')

    # The four least frequent
    # axs[0, 0].plot(hitrates[19]['size'], hitrates[19]['hitrate'])
    # axs[0, 1].plot(hitrates[30]['size'], hitrates[30]['hitrate'])
    # axs[1, 0].plot(hitrates[26]['size'], hitrates[26]['hitrate'])
    # axs[1, 1].plot(hitrates[32]['size'], hitrates[32]['hitrate'])

    fig.suptitle('Hit-rate curves')
    plt.tight_layout()
    plt.show()
