import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import argparse
import sys
sys.path.append('..')
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

# from '1,3-6,9,11' to [1,3,6,7,8,9,11]
def get_range_list(value):
    v = []
    comma_separated = value.split(',')
    for ds in comma_separated:
        dash_separated = ds.split('-')
        if len(dash_separated) == 1:
            v.append(int(dash_separated[0]))
        else:
            v += range(int(dash_separated[0]), int(dash_separated[1])+1)
    return v
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Power Law distribution of the IDs order statistics')
    parser.add_argument('--files','-f', type=str, default=None, required=True)
    parser.add_argument('--feature-indexes','-i', type=range_list, required=True)
    parser.add_argument('--chunk-size','-c', type=int, default=int(1e6))
    parser.add_argument('--drop-nan','-d', action='store_true', default=False)
    parser.add_argument('--gzip','-z', action='store_true', default=False)
    parser.add_argument('--n-chunks','-n', type=int, default=None)
    parser.add_argument('--max-ids','-m', type=int, default=np.inf) 
    parser.add_argument('--until-first','-F', type=int, default=10)
    args = parser.parse_args()

    selected_columns = get_range_list(args.feature_indexes)
    pandas_kwargs = {
        'sep' : '\t',
        'header' : None,
        'usecols' : selected_columns,
        'chunksize' : args.chunk_size,
        'compression' : 'gzip' if args.gzip else None
    }

    # using ChunkStreaming to process value_counts
    cs = ChunkStreaming(
        files=args.files,
        drop_nan=args.drop_nan,
        nchunks=args.n_chunks,
        **pandas_kwargs
    )
    cs.column_mapper = pd.Series.value_counts
    cs.column_feeder = selected_columns
    cs.column_reducer = lambda x,y: pd.concat([x, y]).groupby(level=0).sum()
    cs.index_transformer = lambda x: str(x)
    
    t = time()
    vcs = cs.process_columns()
    t = time() - t
    print(f'Elapsed time\t: {t:.5} sec')

    print('Sorting data ... ', end='', flush=True)
    t = time()
    for index,vc in vcs:
        vc.where(vc > args.until_first, inplace=True)
        vc.dropna(inplace=True)
        vc.sort_values(ascending=False, inplace=True)
    t = time() - t
    print(f'{t:.5} sec')

    print('Selecting data:')
    for index,vc in vcs:
        max_ids = min(args.max_ids, vc.size)
        print(f'Feature {index}: Displaying {max_ids}/{vc.size} IDs')
        vc = vc[0:max_ids]
        plt.plot(range(max_ids), vc, label=f'Feature {index}')

    # saving to numpyz file
    names = dict(vcs)
    outname = 'ustats_' + str(int(time()))
    np.savez(outname, vc, **dict(vcs))

    # plotting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Order statistic')
    plt.ylabel('Frequency')
    plt.xlim(left=1)
    plt.title('Power Law distribution of the IDs order statistics')
    #plt.legend()
    plt.savefig(outname + '.png')
    plt.show()



    