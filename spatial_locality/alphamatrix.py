# Calculates, plots and saves the alpha correlation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from time import time
import sys; sys.path.append('..')
import bigdatatools
from bigdatatools import ChunkStreaming


def alpha_correlation(cardinalities, N, t):
    if type(cardinalities) is not np.ndarray:
        cardinalities = np.array(cardinalities)
    M = min(cardinalities.prod(), N)
    m = cardinalities.max()
    return (M - t) / (M - m)


if __name__ == '__main__':
    parser = bigdatatools.default_parser()
    parser.description = 'Computing alpha correlation of pairs of categorical features'
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

    # extract feature domain cardinalities
    all_cardinalities = {}
    for c in column_selection:
        match_finder = (
            next((
                (vcounts, i)
                for i in range(args.order)
                if index[i] == c
            ), None)
            for index, vcounts in vcs
        )
        vcounts, i = next(
            match for match in match_finder
            if match is not None
        )
        all_cardinalities[c] = vcounts.index.get_level_values(i).nunique()

    # compute all correlations in a numpy matrix
    nsel = len(column_selection)
    A = np.empty(tuple([nsel] * args.order)) # alpha matrix
    A[:] = np.nan

    # filling alpha matrix
    alphas = []
    offset = min(column_selection)
    for index, vcounts in vcs:
        matrix_index = tuple([i - offset for i in index])
        A[matrix_index] = alpha_correlation(
            [all_cardinalities[i] for i in index],
            cs.processed_rows,
            vcounts.size
        )
        alphas.append((index, A[matrix_index]))

    # rearrangin alpha matrix in such a way that correlated features
    # appears one after the other

    # finding a feature reordering according to alpha correlation
    new_order = []
    alphas.sort(reverse=True, key=lambda x: x[1])
    for (f1, f2), _ in alphas:
        if f1 not in new_order:
            new_order.append(f1)
        if f2 not in new_order:
            new_order.append(f2)

    # getting the permutation corresponding to 'new_order' ordering
    permutation = dict()
    for i, fi in enumerate(new_order):
        permutation[fi] = i

    # finally creating and filling B
    B = np.empty(tuple([nsel] * args.order)) # permutation of A
    B[:] = np.nan
    for i in range(nsel):
        for j in range(nsel):
            # if i == j:
            #     pass
            A_location = tuple(sorted(
                [new_order[i] - offset, new_order[j] - offset]
            ))
            B[i, j] = A[A_location]

    t = time() - t
    print(f'Elapsed time\t: {t:.5} sec')

    # saving to file
    timestamp = str(int(time()))
    np.save('alphaA_' + timestamp, A)
    np.save('alphaB_' + timestamp, B)

    # plotting A
    plt.figure(0)
    plt.imshow(A)
    plt.xticks(range(nsel), column_selection)
    plt.yticks(range(nsel), column_selection)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('alphaA_' + timestamp + '.png')

    # plotting B
    plt.figure(1)
    plt.imshow(B)
    plt.xticks(range(nsel), new_order)
    plt.yticks(range(nsel), new_order)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('alphaB_' + timestamp + '.png')

    plt.show()
