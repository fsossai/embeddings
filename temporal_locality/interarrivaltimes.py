import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from scipy import stats
import sys
import argparse


def interarrival_times_of(key, data, feature_index):
    occurrences = np.where(data[feature_index] == key)

    # calculating interarrival times
    a, b = iter(occurrences[0]), iter(occurrences[0])
    next(b, None)
    inter = list(map(lambda t1, t2: t2 - t1, a, b))
    return len(occurrences[0]), inter


def interarrival_times(data, feature_index, order_stat):
    # selecting the order_stat-most frequent ID
    key = list(data[feature_index]
               .replace(np.nan, 0)
               .value_counts()
               .to_dict())[order_stat]
    l, inter = interarrival_times_of(key, data, feature_index)
    return key, l, inter


def group_interarrivals(inter):
    inter.sort()
    xy = list(zip(*[
        (key, len(list(val)))
        for key, val in groupby(sorted(inter))
    ]))
    x = np.array(xy[0])
    y = np.array(xy[1])
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Histograms of interarrival times for an index of a sparse feature'
    )
    parser.add_argument('--feature-index', '-f', type=int, default=14)
    parser.add_argument('--top', '-t', type=int, default=1)
    args = parser.parse_args()

    if args.feature_index not in range(14, 39 + 1):
        print('ERROR: Feature must be in range [14,39]')
        sys.exit(-1)

    reader = pd.read_csv(
        '..\\data\\day_23.gz',
        sep='\t',
        header=None,
        compression='gzip',
        chunksize=int(1e6)
    )
    data = next(reader)

    key, count, inter = interarrival_times(data, args.feature_index, args.top - 1)
    print(f'key: {key}')
    print(f'#occurrences: {count}')
    print(f'Poisson 1/lambda guess\t: {len(data) / count:.15}')

    # creating pairs (interarrival time, how many)
    x, y = group_interarrivals(inter)

    # fitting data with an exponential distribution with SciPy
    mean = np.mean(inter)
    loc, scale = stats.expon.fit(inter)
    lam = 1 / scale
    max_y = np.max(y) / len(inter) * 1.05
    # lam = 1 / mean
    print(f'Emprirical mean\t\t: {mean:.15}')
    print(f'SciPy fit scale\t\t: {scale:.15}')
    print(f'SciPy fit loc\t\t: {loc:.15}')
    print(f'mean - (scale + loc)\t: {mean - (scale + loc)}')

    plt.bar(x, y / len(inter), color=(0.2, 0.4, 0.8, 0.5),
            label=f'{args.top}-th most frequent embedding ID', width=1)

    # plt.plot(x, lam * np.exp(-lam * (x-loc)), 'r-', label='MLE exponential fit')
    plt.plot(x, (1 / mean) * np.exp(-x / mean), 'r-', label='MLE exponential fit')
    plt.plot([mean, mean], [0, max_y], 'b-.', lw=1.3, label='Empirical mean')
    # plt.xlim(left=0.5, right=250)
    plt.title(f'Interarrival times distribution, feature {args.feature_index}')
    plt.xlabel('Interarrival time (discrete units)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
