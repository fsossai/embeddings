import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

import sys; sys.path.append('..')
import bigdatatools
from bigdatatools import ChunkStreaming

if __name__ == '__main__':
    parser = bigdatatools.default_parser()
    parser.description = 'Cardinality of categorical features\' domain'
    parser.add_argument('--files', '-f', type=str, default=None, required=True)
    parser.add_argument('--drop-nan', '-d', action='store_true', default=False)
    args = parser.parse_args()

    column_selection = bigdatatools.get_range_list(args.column_selection)
    pandas_kwargs = {
        'sep': '\t',
        'header': None,
        'usecols': column_selection,
        'chunksize': args.chunk_size,
        'compression': 'gzip' if args.gzip else None
    }

    cs = ChunkStreaming(args.files,
                        drop_nan=args.drop_nan,
                        nchunks=args.n_chunks,
                        parallel=False,
                        **pandas_kwargs)
    cs.column_mapper = pd.Series.value_counts
    cs.column_feeder = column_selection
    cs.column_reducer = lambda x, y: (
            pd.concat([x, y])
            .groupby(level=0)
            .sum()
            .sort_values(ascending=False)
        )

    t = time()
    vcounts = cs.process_columns()
    t = time() - t

    # cumuative distribution of frequencies

    N1, N2 = 4, 5
    fig, axs = plt.subplots(N1, N2)
    sizes = [(i, vc.size) for i, vc in vcounts]
    
    cdf = dict([
        (i, np.cumsum(vc.to_numpy() / vc.sum()))
        for i, vc in vcounts
    ])

    # selecting only the biggest features
    plt_selection = [x for x,y in sorted(sizes, reverse=True, key=lambda x: x[1])]

    sizes = dict([(i, vc.size) for i, vc in vcounts])
    k = 0
    for i in range(N1):
        for j in range(N2):
            sel = plt_selection[k]
            axs[i, j].plot(range(1, sizes[sel]+1), cdf[sel])
            axs[i, j].set(title='Feature ' + str(sel))
            k += 1

    plt.tight_layout()
    plt.show()