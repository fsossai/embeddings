# Some tools to process big files in chunks, with one pass.

import pandas as pd
import numpy as np
import argparse
from time import time


def check_functors(mapper, reducer):
    if mapper is None:
        raise Exception('empty mapper')
    if reducer is None:
        raise Exception('empty reducer')


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
    if value is None:
        return None
    v = []
    comma_separated = value.split(',')
    for ds in comma_separated:
        dash_separated = ds.split('-')
        if len(dash_separated) == 1:
            v.append(int(dash_separated[0]))
        else:
            v += range(int(dash_separated[0]), int(dash_separated[1]) + 1)
    return v


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-size', '-c', type=int, default=int(1e6))
    parser.add_argument('--n-chunks', '-n', type=int, default=None)
    parser.add_argument('--gzip', '-z', action='store_true', default=False)
    parser.add_argument('--column-selection', '-S', type=range_list, default=None)
    return parser


class ChunkStreaming:
    def __init__(self, files, drop_nan=False, log=True, nchunks=np.inf, **pandas_args):
        if '*' in files:
            from glob import glob
            self.files = glob(files)
        else:
            self.files = [files]
        self.column_mapper = None
        self.column_feeder = None
        self.column_reducer = None
        self.chunk_mapper = None
        self.chunk_reducer = None
        self.index_transformer = lambda x: x
        self.drop_nan = drop_nan
        self.pandas_args = pandas_args
        self.processed_rows = 0
        self.log = log
        self.nchunks = nchunks
        self.__chunk_counter = 0

    def chunk_gen(self, pandas_args=None):
        if pandas_args is None:
            pandas_args = self.pandas_args
        for file in self.files:
            reader = pd.read_csv(file, **pandas_args)
            for chunk in reader:
                if self.__chunk_counter >= self.nchunks:
                    return
                if self.log:
                    print(
                        f'File\t: \'{file}\',\t' +
                        f'chunk number\t: {self.__chunk_counter + 1}',
                        end='', flush=True
                    )
                    if self.nchunks is np.inf:
                        print()
                    else:
                        print(f'/{self.nchunks}')
                if self.drop_nan:
                    chunk.dropna(inplace=True)
                else:
                    chunk.replace(np.nan, '0', inplace=True)
                self.__chunk_counter += 1
                yield chunk
                self.processed_rows += len(chunk)
                if self.log:
                    print()

    def foreach_column(self, mapper, feeder, reducer):
        self.processed_rows = 0
        check_functors(mapper, reducer)
        chunk = self.chunk_gen()
        data = next(chunk)

        if feeder is None:
            feeder = data.columns

        def map_gen(d):
            for work in feeder:
                if self.log:
                    print(f'\r mapping {work} ... ', end='', flush=True)
                if type(work) in [int, list]:
                    yield self.index_transformer(work), mapper(d[work])
                elif type(work) is tuple:
                    yield self.index_transformer(work), mapper(d[list(work)])
                else:
                    raise Exception('feeder type not supported')

        mapped0 = list(map_gen(data))
        if self.log: print()

        data = next(chunk, None)
        while data is not None:
            t = time()
            mapped1 = list(map_gen(data))
            if self.log:
                print('reducing ... ', end='', flush=True)
            mapped0 = [
                (i, reducer(m0, m1))
                for (i, m0), (_, m1) in zip(mapped0, mapped1)
            ]
            t = time() - t
            if self.log:
                print(f'{t:.5} sec')
            data = next(chunk, None)

        return list(mapped0)

    def foreach_chunk(self, mapper, reducer):
        self.processed_rows = 0
        check_functors(mapper, reducer)

        chunk = self.chunk_gen()
        data = next(chunk)

        mapped0 = mapper(data)

        data = next(chunk, None)
        while data is not None:
            mapped1 = mapper(data)
            mapped0 = [
                reducer(m0, m1)
                for m0, m1 in zip(mapped0, mapped1)
            ]
            data = next(chunk, None)

        return list(mapped0)

    def process_columns(self):
        return self.foreach_column(
            self.column_mapper,
            self.column_feeder,
            self.column_reducer
        )

    def process_chunks(self):
        return self.foreach_chunk(self.chunk_mapper, self.chunk_reducer)
