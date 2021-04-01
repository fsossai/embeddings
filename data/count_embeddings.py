import pandas as pd
import matplotlib.pyplot as plt
from time import time

import sys; sys.path.append('..')
import bigdatatools
from bigdatatools import ChunkStreaming
from plot_counts import setup_histogram

if __name__ == '__main__':
    parser = bigdatatools.default_parser()
    parser.description = 'Cardinality of categorical features\' domain'
    parser.add_argument('--files', '-f', type=str, default=None, required=True)
    parser.add_argument('--drop-nan', '-d', action='store_true', default=False)
    parser.add_argument('--top', '-t', type=int, default=None)
    parser.add_argument('--highlight', '-H', type=int, default=0)
    parser.add_argument('--linear-plot', '-l', action='store_true', default=False)
    parser.add_argument('--reverse-order', '-r', action='store_true', default=False)
    parser.add_argument('--out-name', '-o', type=str, default=None, required=False)
    
    args = parser.parse_args()

    timestamp = str(time())
    out_name = args.out_name or f'counts_{timestamp}'

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
    cs.column_reducer = lambda x, y: pd.concat([x, y]).groupby(level=0).sum()

    t = time()
    vcounts = cs.process_columns()

    # creating tuples to be plotted
    cardinalities = [
        (
            i,
            vcount.size,
            (vcount <= args.highlight).sum()
        )
        for i, vcount in vcounts
    ]

    print(*cardinalities, sep='\n')
    print('\nSorted:')
    cardinalities.sort(key=lambda x: x[1], reverse=not args.reverse_order)
    t = time() - t

    # exporting counts to csv file
    pd.DataFrame(
        cardinalities,
        columns=['table', 'nunique', 'highlighted']
    ).to_csv(out_name + '.csv', index=False)

    if args.top is not None:
        cardinalities = cardinalities[:args.top]
    print(*cardinalities, sep='\n')
    print(f'\nElapsed time: {t} sec')

    nsel = args.top if args.top is not None else len(column_selection)
    column_selection = [x for x, y in vcounts]
    xy = list(zip(*cardinalities))

    setup_histogram(
        nuniques=xy[1],
        highlighted=xy[2],
        highlight=args.highlight,
        ticks=xy[0],
        linear_plot=args.linear_plot,
        top=args.top
    )
    plt.savefig(out_name + '.png')
    plt.show()

# Command line example:
# python count_embeddings.py -f *.gz -z -c 1000000 -n 1 -S 14-39 -l -t 10 -H 1
