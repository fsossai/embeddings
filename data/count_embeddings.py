import pandas as pd
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
                        parallel=True,
                        **pandas_kwargs)
    cs.column_mapper = pd.Series.value_counts
    cs.column_feeder = column_selection
    cs.column_reducer = lambda x, y: pd.concat([x, y]).groupby(level=0).sum()

    t = time()
    vcounts = cs.process_columns()
    t = time() - t

    # creating tuples to be plotted
    cardinalities = [(i, vcount.size) for i, vcount in vcounts]
    print(f'{t} sec')
    print(*cardinalities, sep='\n')
    print('\nSorted:')
    cardinalities.sort(key=lambda x: x[1], reverse=True)
    print(*cardinalities, sep='\n')

    # plotting
    column_selection = [x for x, y in vcounts]
    xy = list(zip(*cardinalities))
    plt.bar(cs.latest_columns, xy[1], log=True)
    plt.xticks(cs.latest_columns, xy[0], rotation=70)
    plt.xlabel('Feature index')
    plt.ylabel('Number of unique IDs')
    plt.title("Cardinality of the features' domain")
    plt.tight_layout()
    plt.show()
