# Some tools to process big files in chunks, with one pass.

import pandas as pd
import numpy as np
from time import time

class ChunkStreaming:
    def __init__(self, files, drop_nan, log=True, **pandas_args):
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
        self.drop_nan = drop_nan
        self.pandas_args = pandas_args
        self.processed_rows = 0
        self.log = log

    def __check_func(self, mapper, reducer):
        if mapper is None:
            raise Exception('empty mapper')
        if reducer is None:
            raise Exception('empty reducer')

    def __chunk_generator(self):
        for file in self.files:
            if self.log:
                print(f'\nCurrent file\t: \'{file}\'')
            reader = pd.read_csv(file, **self.pandas_args)
            for chunk in reader:
                if self.drop_nan:
                    chunk.dropna(inplace=True)
                else:
                    chunk.replace(np.nan, '0', inplace=True)
                yield chunk
                self.processed_rows += len(chunk)

    def foreach_column(self, mapper, feeder, reducer):
        self.processed_rows = 0
        self.__check_func(mapper, reducer)
        chunk = self.__chunk_generator()
        data = next(chunk)

        if feeder is None:
            feeder = data.columns

        def map_gen(d):
            for work in feeder:
                if self.log:
                    print(f'\r mapping {work} ... ', end='', flush=True)
                if type(work) in [int, list]:
                    yield ( work, mapper(d[work]) )
                elif type(work) is tuple:
                    yield ( work, mapper(d[list(work)]) )
                else:
                    raise Exception('feeder type not supported')

        if self.log:
            print()

        mapped0 = list(map_gen(data))

        data = next(chunk, None)
        while (data is not None):
            t = time()
            mapped1 = list(map_gen(data))
            if self.log:
                print('reducing ... ', end='', flush=True)
            mapped0 = [
                ( i, reducer(m0, m1) )
                for (i,m0),(_,m1) in zip(mapped0, mapped1)
            ]
            t = time() - t
            if self.log:
                print(f'{t:.5} sec')
            data = next(chunk, None)
        
        return list(mapped0)
    
    def foreach_chunk(self, mapper, reducer):
        self.processed_rows = 0
        self.__check_func(mapper, reducer)

        chunk = self.__chunk_generator()
        data = next(chunk)

        mapped0 = mapper(data)

        data = next(chunk, None)
        while (data is not None):
            mapped1 = mapper(data)
            mapped0 = [
                reducer(m0, m1)
                for m0,m1 in zip(mapped0, mapped1)
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

    