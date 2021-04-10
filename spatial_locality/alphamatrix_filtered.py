# Calculates, plots and saves the alpha correlation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from time import time
import sys; sys.path.append('..')
import bigdatatools
from bigdatatools import ChunkStreaming

cardinalities = dict()
index_to_drop = dict()
u = []
filter = 1

def get_index_to_drop(pair):
    index, vcx = pair
    for level in [0, 1]:
        col = index[level]
        if col in index_to_drop:
            continue
        grouped = vcx.groupby(level=level)
        index_to_drop[col] = [
            i
            for i, s in grouped
            if s.sum() <= filter
        ]
        u.append((col, grouped.size().size))
    return index_to_drop[index[0]], index_to_drop[index[1]]

def alpha_correlation(pair):
    (i1, i2), vcx = pair
    to_drop1, to_drop2 = get_index_to_drop(pair)
    total_drops = len(to_drop1) + len(to_drop2)
    vcx_f = (
        vcx
        .drop(level=0, labels=to_drop1, errors='ignore')
        .drop(level=1, labels=to_drop2, errors='ignore')
    )
    u1 = vcx_f.index.get_level_values(0).nunique()
    u2 = vcx_f.index.get_level_values(1).nunique()
    N = vcx_f.sum()
    t = vcx_f.size
    M = min(u1 * u2, N)
    m = max(u1, u2)
    return (M - t) / (M - m)

if __name__ == '__main__':
    parser = bigdatatools.default_parser()
    parser.description = 'Computing alpha correlation of pairs of categorical features'
    parser.add_argument('--files', '-f', type=str, default=None, required=True)
    parser.add_argument('--order', '-o', type=int, default=2)
    parser.add_argument('--filter', '-l', type=int, default=0)
    parser.add_argument('--save', '-s', action="store_true", default=False)
    args = parser.parse_args()

    filter = args.filter
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

    # A
    # compute all correlations in a numpy matrix
    nsel = len(column_selection)
    A = np.empty(tuple([nsel] * args.order)) # alpha matrix
    A[:] = np.nan

    # filling alpha matrix
    alphas = []
    offset = min(column_selection)
    for pair in vcs:
        index, vcx = pair
        matrix_index = tuple([i - offset for i in index])
        A[matrix_index] = alpha_correlation(pair)
        A[tuple(reversed(matrix_index))] = A[matrix_index]
        alphas.append((index, A[matrix_index]))

    # B
    # rearranging alpha matrix in such a way that correlated features
    # appears one after the other

    # finding a feature reordering according to alpha correlation
    B_order = []
    alphas.sort(reverse=True, key=lambda x: x[1])
    for (f1, f2), _ in alphas:
        if f1 not in B_order:
            B_order.append(f1)
        if f2 not in B_order:
            B_order.append(f2)

    # finally creating and filling B
    B = np.empty(tuple([nsel] * args.order)) # permutation of A
    B[:] = np.nan
    for i in range(nsel):
        for j in range(nsel):
            A_location = tuple(sorted(
                [B_order[i] - offset, B_order[j] - offset]
            ))
            B[i, j] = A[A_location]

    # C
    # rearrangement based on the biggest ones
    C_order = [i for i, _ in sorted(u, reverse=True, key=lambda x: x[1])]

    # filling C
    C = np.empty(tuple([nsel] * args.order)) # permutation of A
    C[:] = np.nan
    for i in range(nsel):
        for j in range(nsel):
            A_location = tuple(sorted(
                [C_order[i] - offset, C_order[j] - offset]
            ))
            C[i, j] = A[A_location]

    # D
    # custom reordering
    D_order = [33,14,35,23,34,24,36,15,28,27,21,25,16,18,37,20,17,31,29,38,22,30,39,32,19,26]

    # filling D
    D = np.empty(tuple([nsel] * args.order)) # permutation of A
    D[:] = np.nan
    for i in range(nsel):
        for j in range(nsel):
            A_location = tuple(sorted(
                [D_order[i] - offset, D_order[j] - offset]
            ))
            D[i, j] = A[A_location]

    t = time() - t
    print(f'Elapsed time\t: {t:.5} sec')

    # saving to file
    timestamp = str(int(time()))
    if args.save:
        np.save('alpha_f' + timestamp + '_A', A)
        np.save('alpha_f' + timestamp + '_B', B)
        np.save('alpha_f' + timestamp + '_C', C)
        np.save('alpha_f' + timestamp + '_D', D)

    # plotting A
    plt.figure(0)
    plt.title('Alpha matrix - No reordering')
    plt.imshow(A)
    plt.xticks(range(nsel), column_selection, rotation=90)
    plt.yticks(range(nsel), column_selection)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    if args.save:
        plt.savefig('alpha_f' + timestamp + '_A.png')

    # plotting B
    plt.figure(1)
    plt.title('Alpha matrix - Correlation-based reordering')
    plt.imshow(B)
    plt.xticks(range(nsel), B_order, rotation=90)
    plt.yticks(range(nsel), B_order)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    if args.save:
        plt.savefig('alpha_f' + timestamp + '_B.png')

    # plotting C
    plt.figure(2)
    plt.title('Alpha matrix - Sorted by table size')
    plt.imshow(C)
    plt.xticks(range(nsel), C_order, rotation=90)
    plt.yticks(range(nsel), C_order)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    if args.save:
        plt.savefig('alpha_f' + timestamp + '_C.png')

    # plotting D
    plt.figure(3)
    plt.title('Alpha matrix - Custom reordering')
    plt.imshow(D)
    plt.xticks(range(nsel), D_order, rotation=90)
    plt.yticks(range(nsel), D_order)
    plt.colorbar()
    plt.grid(True)
    plt.tight_layout()
    if args.save:
        plt.savefig('alpha_f' + timestamp + '_D.png')

    plt.show()
