# this script is just a draft

import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from time import time

def get_pval(mu, delta):
    return np.exp(-mu * (delta * delta / (2 + 2 / 3 * delta)))

def get_mu(vcounts1, vcounts2, N):
    mu = len(vcounts1.index) * len(vcounts2.index)
    N2 = N * N
    for i1 in vcounts1.index:
        for i2 in vcounts2.index:
            mu -= (1 - vcounts1[i1] * vcounts2[i2] / N2) ** N
    return mu

if __name__ == '__main__':
    data = pd.read_csv('..\\data\\day_100k.csv', sep='\t', header=None)
    print('Loaded')
    data = data[0:100]
    N = len(data)
    column_selection = list(range(14,39+1))
    pairs = combinations(column_selection, r=2)

    t = time()

    single_vcounts = dict()
    nsel = len(column_selection)
    A = np.empty((nsel,nsel), dtype=np.float32)
    A[:] = np.nan
    offset = min(column_selection)

    for i1, i2 in pairs:
        if i1 not in single_vcounts:
            single_vcounts[i1] = data[i1].value_counts()
        if i2 not in single_vcounts:
            single_vcounts[i2] = data[i2].value_counts()
        vcounts1 = single_vcounts[i1]
        vcounts2 = single_vcounts[i2]
        mu_e = data[[i1,i2]].value_counts().count()
        mu = get_mu(vcounts1, vcounts2, N)
        delta = 1 - mu_e / mu
        pvalue = get_pval(mu, delta)
        if mu_e > mu:
            print('!', end='')
        print(f'Pvalue {(i1,i2)}\t\t: {pvalue}')
        A[i1-offset,i2-offset] = 1-pvalue

    t = time() - t
    print(f'Elapsed time\t: {t:.5} sec')

    # saving to file
    outname = 'chern_' + str(int(time()))
    np.save(outname, A)
    # plotting image
    plt.imshow(A)
    plt.xticks(range(len(column_selection)), column_selection)
    plt.yticks(range(len(column_selection)), column_selection)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outname + '.png')
    plt.show()