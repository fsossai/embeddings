import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob
from time import time

import sys; sys.path.append('..')
import bigdatatools
from bigdatatools import ChunkStreaming

if __name__ == '__main__':
    parser = bigdatatools.default_parser()
    parser.description = 'Cumulative frequencies compared to hit-rate of LRU'
    parser.add_argument('--files', '-f', type=str, default=None, required=True)
    parser.add_argument('--hitrates', '-H', type=str, default=None, required=True)
    
    args = parser.parse_args()

    all_files = glob(args.hitrates)
    if len(all_files) == 0:
        print('ERROR: hitrates files not found')
        sys.exit(-1)
  
    # importing into DataFrame all CSV files of the hitrates
    hitrates = dict()
    for file in all_files:
        index = int(re.findall(r'\d+', file)[-1])
        hitrates[index] = pd.read_csv(file)

    # loading dataset for cumfreq computation
    column_selection = bigdatatools.get_range_list(args.column_selection)
    pandas_kwargs = {
        'sep': '\t',
        'header': None,
        'usecols': column_selection,
        'chunksize': args.chunk_size,
        'compression': 'gzip' if args.gzip else None
    }

    cs = ChunkStreaming(args.files,
                        drop_nan=False,
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
    
    # cumulative distribution of frequencies
    sizes = [(i, vc.size) for i, vc in vcounts]
    cdf = dict([
        (i, np.cumsum(vc.to_numpy() / vc.sum()))
        for i, vc in vcounts
    ])
    t = time() - t    
    print(f'Elapsed time: {t} sec')

    # plt_selection = [x for x,y in sorted(sizes, reverse=True, key=lambda x: x[1])]

    nsel = len(column_selection)
    fig, axs = plt.subplots(2, nsel)
    sizes = dict([(i, vc.size) for i, vc in vcounts])

    for i, feat in enumerate(column_selection):
        axs[0, i].plot(range(1, sizes[feat]+1), cdf[feat])
        axs[0, i].set(ylim=[0, 1])
        axs[0, i].set(title=f'Feature {feat}')
        axs[1, i].plot(hitrates[feat]['size'], hitrates[feat]['hitrate'])
        axs[1, i].set(ylim=[0, 1])
        
    axs[0, 0].set(ylabel='Cumulative frequency')
    axs[1, 0].set(ylabel='Hit-rate')
    plt.tight_layout()
    plt.show()

# Command line example:
# python cumfreq_vs_hitrate.py -c 1000000 -n 1 -z -f "..\data\*.gz" -H hitrates_csv\LRU* -S 14,25,16,37