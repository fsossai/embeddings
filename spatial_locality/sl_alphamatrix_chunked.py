# Calulates, plots and saves the alpha correlation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from time import time
import argparse

import sys
sys.path.append('..')
from bigdatatools import ChunkStreaming

def alpha_correlation(cardinalities, N, t):
    if type(cardinalities) is not np.ndarray:
        cardinalities = np.array(cardinalities)
    M = min(cardinalities.prod(), N)
    m = cardinalities.max()
    return (M-t) / (M-m)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computing alpha correlation of pairs of categorical features')
    parser.add_argument('--files','-f', type=str, default=None, required=True)
    parser.add_argument('--chunk-size','-c', type=int, default=int(1e6))
    parser.add_argument('--drop-nan','-d', action='store_true', default=False)
    parser.add_argument('--gzip','-z', action='store_true', default=False)
    parser.add_argument('--order','-o', type=int, default=2)
    args = parser.parse_args()

    selected_columns = range(14, 14+26)
    kwargs = {
        'sep' : '\t',
        'header' : None,
        'usecols' : selected_columns,
        'chunksize' : args.chunk_size,
        'compression' : 'gzip' if args.gzip else None
    }
 
    def reducer(x,y):
        return pd.concat([x,y]).groupby(
            level=list(range(args.order))
        ).sum()

    t = time()
    # counting unique r-tuples of IDs in each r-tuple of columns
    cs = ChunkStreaming(args.files, args.drop_nan, **kwargs)
    cs.column_mapper = pd.DataFrame.value_counts
    cs.column_feeder = list(combinations(selected_columns, r=args.order))
    cs.column_reducer = reducer
    vcs = cs.process_columns()
    
    # extract feature domain cardinalities
    all_cardinalities = {}
    for c in selected_columns:
        match_finder = (
            next((
                (vcounts,i)
                for i in range(args.order)
                if index[i] == c
            ), None)
            for index,vcounts in vcs
        )
        vcounts,i = next(
            match for match in match_finder
            if match is not None
        )
        all_cardinalities[c] = vcounts.index.get_level_values(i).nunique()

    # compute all correlations in a numpy matrix
    nsel = len(selected_columns)
    A = np.empty( tuple([nsel] * args.order) )
    A[:] = np.nan
    offset = min(selected_columns)
    for index,vcounts in vcs:
        matrix_index = tuple([i-offset for i in index])
        A[matrix_index] = alpha_correlation(
            [all_cardinalities[i] for i in index],
            cs.processed_rows,
            vcounts.size
        )

    t = time() - t
    print(f'Elapsed time\t: {t:.5} sec')
    
    # saving to file
    outname = 'alpha_' + str(int(time()))
    np.save(outname, A)

    # plotting image
    plt.imshow(A)
    plt.xticks(range(len(selected_columns)), selected_columns)
    plt.yticks(range(len(selected_columns)), selected_columns)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outname + '.png')
    plt.show()



