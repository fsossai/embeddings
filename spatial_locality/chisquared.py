# Implementing Pearson's Chi-squared test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from time import time
from scipy.stats import chisquare, chi2_contingency
import sys; sys.path.append('..')
import bigdatatools
from bigdatatools import ChunkStreaming

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1) * (r-1)) / (n-1))
    rcorr = r - ((r-1) ** 2) / (n-1)
    kcorr = k - ((k-1) ** 2) / (n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

if __name__ == '__main__':
    parser = bigdatatools.default_parser()
    parser.description = "Performing the Pearson's Chi-squared test" \
            " on pairs of categorical features"
    parser.add_argument('--files', '-f', type=str, default=None, required=True)
    parser.add_argument('--order', '-o', type=int, default=2)
    args = parser.parse_args()

    column_selection = bigdatatools.get_range_list(args.column_selection)
    kwargs = {
        'sep': '\t',
        'header': None,
        'usecols': column_selection,
        'chunksize': args.chunk_size,
        'compression': 'gzip' if args.gzip else None
    }


    def reducer(x, y):
        return pd.concat([x, y]).groupby(
            level=list(range(args.order))
        ).sum()


    t = time()
    # counting unique r-tuples of IDs in each r-tuple of columns
    cs = ChunkStreaming(args.files, nchunks=args.n_chunks, **kwargs)
    cs.column_mapper = pd.DataFrame.value_counts
    cs.column_feeder = list(combinations(column_selection, r=args.order))
    cs.column_reducer = reducer
    vcs = cs.process_columns()
    N = cs.processed_rows

    # matrix initialization
    nsel = len(column_selection)
    A = np.empty(tuple([nsel] * args.order))
    A[:] = np.nan
    offset = min(column_selection)

    # obtaining the number of unique values for each column
    col_vcounts = dict()
    for index, vcounts in vcs:
        for i,col in enumerate(index):
            if col not in col_vcounts:
                col_vcounts[col] = dict([
                    (x, y.sum()) for x,y in vcounts.groupby(level=i)
                ])

    # double-checking, did we obtain values for all columns?
    for col in column_selection:
        if col not in col_vcounts:
            print(f'Error: no results for column {col}!')
            sys.exit(-1)

    for index, vcounts in vcs:
        order = len(index)
        assert N == vcounts.sum() # might be false when excluding nan
        nuniques = len(vcounts)
        f_obs = vcounts.to_numpy()
        # every index of the vcounts series is listed in 'elem'
        elem = list(zip(*vcounts.index))
        # seeking the number of occurrences of each index appearing in vcounts
        occurrences = np.empty((order, nuniques), dtype=np.float32)
        for i, col in enumerate(index):
            # substituting an embedding with its number of occurrences
            occurrences[i,:] = np.array([
                col_vcounts[col][x] for x in elem[i]
            ])
        d = 1 / (N ** (order - 1))
        f_exp = occurrences.prod(axis=0) * d
        matrix_index = tuple([i - offset for i in index])
        A[matrix_index] = chisquare(f_obs, f_exp).pvalue

    t = time() - t
    print(f'Elapsed time\t: {t:.5} sec')

    # saving to file
    outname = 'chisq_' + str(int(time()))
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
