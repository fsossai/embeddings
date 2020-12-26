import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from time import time

import sys
sys.path.append('..')
from bigdatatools import ChunkStreaming

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cardinality of categorical features\' domain')
    parser.add_argument('--files','-f', type=str, default=None, required=True)
    parser.add_argument('--chunk-size','-c', type=int, default=int(1e6))
    parser.add_argument('--drop-nan','-d', action='store_true', default=False)
    parser.add_argument('--gzip','-z', action='store_true', default=False)
    args = parser.parse_args()

    selected_columns = range(14, 14+26)
    
    kwargs = {
        'sep' : '\t',
        'header' : None,
        'usecols' : selected_columns,
        'chunksize' : args.chunk_size,
        'compression' : 'gzip' if args.gzip else None
    }

    cs = ChunkStreaming(args.files, args.drop_nan, **kwargs)
    cs.column_mapper = pd.Series.value_counts
    cs.column_feeder = selected_columns
    cs.column_reducer = lambda x,y: pd.concat([x, y]).groupby(level=0).sum()
    
    t = time()
    vcs = cs.process_columns()
    t = time() - t

    # creating tuples to be plotted
    cardinalities = [(i,vcounts.size) for i,vcounts in vcs]
    print(f'{t} sec')
    print(*cardinalities, sep='\n')
    print('\nSorted:')
    cardinalities.sort(key=lambda x: x[1], reverse=True)
    print(*cardinalities, sep='\n')

    # plotting
    xy = list(zip(*cardinalities))
    plt.bar(selected_columns, xy[1], log=True)
    plt.xticks(selected_columns, xy[0], rotation=70)
    plt.xlabel('Feature index')
    plt.ylabel('Number of unique IDs')
    plt.title("Cardinality of the features' domain")
    plt.tight_layout()
    plt.plot()
    plt.show()

    