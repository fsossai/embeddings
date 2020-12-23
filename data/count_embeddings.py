import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from time import time
from glob import glob

def chunked_vcounts(**kwargs):
    chunk = chunk_generator(**kwargs)
    data = next(chunk)
    vcs0 = [data[col].value_counts() for col in data.columns]

    data = next(chunk, None)
    while (data is not None):
        vcs1 = [data[col].value_counts() for col in data.columns]
        vcs0 = [
            pd.concat([first, second]).groupby(level=0).sum()
            for first,second in zip(vcs0, vcs1)
        ]
        data = next(chunk, None)
    
    return vcs0

def chunk_generator(file_names, keep_nan, **kwargs):
    for file in file_names:
        reader = pd.read_csv(file, **kwargs)
        for chunk in reader:
            if keep_nan:
                yield chunk.replace(np.nan, '0')
            else:
                yield chunk

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cardinality of categorical features\' domain')
    parser.add_argument('--files','-f', type=str, default=None, required=True)
    parser.add_argument('--chunk-size','-c', type=int, default=int(1e6))
    parser.add_argument('--keep-nan','-k', action='store_true', default=False)
    parser.add_argument('--gzip','-z', action='store_true', default=False)
    args = parser.parse_args()

    selected_columns = range(14, 14+26)
    
    kwargs = {
        'file_names' : glob(args.files),
        'keep_nan' : args.keep_nan,
        'sep' : '\t',
        'header' : None,
        'usecols' : selected_columns,
        'chunksize' : args.chunk_size,
        'compression' : 'gzip' if args.gzip else None
    }

    vcs = chunked_vcounts(**kwargs)

    cardinalities = [(i,counts.size) for i,counts in zip(selected_columns,vcs)]
    t = time() - t
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
    plt.plot()
    plt.show()

    