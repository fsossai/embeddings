import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import argparse
import sys; sys.path.append('..')
import bigdatatools
from bigdatatools import ChunkStreaming


def range_list(value):
    comma_separated = value.split(',')
    for ds in comma_separated:
        dash_separated = ds.split('-')
        for i in dash_separated:
            try:
                int(i)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f'{value} is not a valid dash separated list of ints\n' +
                    'valid range list is, for example, 1,3-6,9,11'
                )
    return value


if __name__ == '__main__':
    parser = bigdatatools.default_parser()
    parser.description='Power Law distribution of the IDs order statistics'
    parser.add_argument('--files', '-f', type=str, default=None, required=True)
    parser.add_argument('--drop-nan', '-d', action='store_true', default=False)
    parser.add_argument('--max-ids', '-m', type=int, default=np.inf)
    parser.add_argument('--until-first', '-F', type=int, default=10)
    args = parser.parse_args()

    column_selection = bigdatatools.get_range_list(args.column_selection)
    pandas_kwargs = {
        'sep': '\t',
        'header': None,
        'usecols': column_selection,
        'chunksize': args.chunk_size,
        'compression': 'gzip' if args.gzip else None
    }

    # using ChunkStreaming to process value_counts
    cs = ChunkStreaming(
        files=args.files,
        drop_nan=args.drop_nan,
        nchunks=args.n_chunks,
        **pandas_kwargs
    )
    cs.column_mapper = pd.Series.value_counts
    cs.column_feeder = column_selection
    cs.column_reducer = lambda x, y: pd.concat([x, y]).groupby(level=0).sum()
    cs.index_transformer = lambda x: str(x)

    t = time()
    vcounts = cs.process_columns()
    t = time() - t
    print(f'Elapsed time\t: {t:.5} sec')

    print('Sorting data ... ', end='', flush=True)
    t = time()
    for index, vc in vcounts:
        vc.where(vc > args.until_first, inplace=True)
        vc.dropna(inplace=True)
        vc.sort_values(ascending=False, inplace=True)
    t = time() - t
    print(f'{t:.5} sec')

    print('Selecting data:')
    for index, vc in vcounts:
        max_ids = min(args.max_ids, vc.size)
        print(f'Feature {index}: Displaying {max_ids}/{vc.size} IDs')
        vc = vc[0:max_ids]
        plt.plot(range(max_ids), vc, label=f'Feature {index}')

    # saving to numpyz file
    names = dict(vcounts)
    outname = 'ustats_' + str(int(time()))
    np.savez(outname, **dict(vcounts))

    # plotting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Order statistic')
    plt.ylabel('Frequency')
    plt.xlim(left=1)
    plt.title('Power Law distribution of the IDs order statistics')
    # plt.legend()
    plt.savefig(outname + '.png')
    plt.show()
