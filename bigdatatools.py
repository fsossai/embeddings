# Some tools to process big files in chunks, with one pass.

import pandas as pd
import numpy as np
import argparse
from time import time
from multiprocessing.pool import ThreadPool
from psutil import cpu_count


def check_functors(mapper, reducer):
    if mapper is None:
        raise Exception('empty mapper')
    if reducer is None:
        raise Exception('empty reducer')


def default_index_transformer(x):
    return x


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
    def __init__(self,
                 files,
                 drop_nan=False,
                 log=True,
                 nchunks=np.inf,
                 parallel=True,
                 column_mapper=None,
                 column_feeder=None,
                 column_reducer=None,
                 chunk_mapper=None,
                 chunk_reducer=None,
                 index_transformer=None,
                 **pandas_args):
        if '*' in files:
            from glob import glob
            self.files = glob(files)
        else:
            self.files = [files]
        self.column_mapper = column_mapper
        self.column_feeder = column_feeder
        self.column_reducer = column_reducer
        self.chunk_mapper = chunk_mapper
        self.chunk_reducer = chunk_reducer
        self.index_transformer = index_transformer or (lambda x: x)
        self.drop_nan = drop_nan
        self.pandas_args = pandas_args
        self.processed_rows = 0
        self.log = log
        self.nchunks = nchunks
        self.__chunk_counter = 0
        self.parallel = parallel
        self.executors = cpu_count(logical=False) if parallel else 1
        self.latest_columns = None
        if 'chunksize' not in pandas_args:
            raise Exception('ChunkStreaming requires \'chunksize\' to be set')

    def chunk_gen(self, pandas_args=None):
        if pandas_args is None:
            pandas_args = self.pandas_args
        for file in self.files:
            reader = pd.read_csv(file, **pandas_args)
            for chunk in reader:
                if self.__chunk_counter >= self.nchunks:
                    return
                self.log_print(
                    f'File\t: \'{file}\',\t' +
                    f'chunk number\t: {self.__chunk_counter + 1}',
                    end='', flush=True
                )
                self.log_print(
                    f'/{self.nchunks}' if self.nchunks is not np.inf
                    else ''
                )
                if self.drop_nan:
                    chunk.dropna(inplace=True)
                else:
                    chunk.replace(np.nan, '0', inplace=True)
                self.__chunk_counter += 1
                yield chunk
                self.processed_rows += len(chunk)
                self.log_print('')

    def __mapper_function(self, pair):
        work, data = pair
        if type(work) is tuple:
            work = list(work)
        return self.index_transformer(work), self.column_mapper(data[work])

    def __reducer_function(self, pairs):
        (work0, data0), (work1, data1) = pairs
        assert work0 == work1, "Attempted to reduce incompatible mapped data"
        return work0, self.column_reducer(data0, data1)

    def process_columns(self):
        self.processed_rows = 0
        check_functors(self.column_mapper, self.column_reducer)

        chunks = self.chunk_gen()
        data = next(chunks)
        feeder = self.column_feeder or data.columns
        self.latest_columns = data.columns
        chops = len(feeder) // self.executors

        def work_gen():
            for work in feeder:
                yield work, data

        # First Mapping phase, this does not require a reduce phase
        self.log_print('mapping\t\t... ', end='', flush=True)
        t = time()
        if self.parallel:
            with ThreadPool(self.executors) as pool:
                mapped0 = list(pool.map(
                    self.__mapper_function, work_gen(), chops))
        else:
            mapped0 = list(map(self.__mapper_function, work_gen()))
        t = time() - t
        self.log_print(f'{t:.5} sec')

        data = next(chunks, None)
        while data is not None:
            # Mapping phase
            self.log_print('mapping\t\t... ', end='', flush=True)
            t = time()
            if self.parallel:
                with ThreadPool(self.executors) as pool:
                    mapped1 = list(pool.map(
                        self.__mapper_function, work_gen(), chops))
            else:
                mapped1 = list(map(self.__mapper_function, work_gen()))
            t = time() - t
            self.log_print(f'{t:.5} sec')

            # Reducing phase
            self.log_print('reducing\t... ', end='', flush=True)
            t = time()
            with ThreadPool(self.executors) as pool:
                mapped0 = list(pool.map(
                    self.__reducer_function,
                    zip(mapped0, mapped1),
                    chops
                ))
            t = time() - t
            self.log_print(f'{t:.5} sec')
            data = next(chunks, None)

        return list(mapped0)

    def foreach_chunk(self, mapper, reducer):
        self.processed_rows = 0
        check_functors(mapper, reducer)

        chunks = self.chunk_gen()
        data = next(chunks)

        mapped0 = mapper(data)

        data = next(chunks, None)
        while data is not None:
            mapped1 = mapper(data)
            mapped0 = [
                reducer(m0, m1)
                for m0, m1 in zip(mapped0, mapped1)
            ]
            data = next(chunks, None)

        return list(mapped0)

    def process_chunks(self):
        return self.foreach_chunk(self.chunk_mapper, self.chunk_reducer)

    def log_print(self, *args, **kwargs):
        if self.log:
            print(*args, **kwargs)
