# Some tools to process big files in chunks, with one pass.

import pandas as pd
import numpy as np

class ChunkStreaming:
    def __init__(self, files, keep_nan, **pandas_args):
        if '*' in files:
            from glob import glob
            self.files = glob(files)
        else:
            self.files = [files]
        self.column_mapper = None
        self.column_reducer = None
        self.chunk_mapper = None
        self.chunk_reducer = None
        self.keep_nan = keep_nan
        self.pandas_args = pandas_args

    def __check_func(self, mapper, reducer):
        if mapper is None:
            raise Exception('empty mapper')
        if reducer is None:
            raise Exception('empty reducer')

    def __chunk_generator(self):
        for file in self.files:
            reader = pd.read_csv(file, **self.pandas_args)
            for chunk in reader:
                if self.keep_nan:
                    yield chunk.replace(np.nan, '0')
                else:
                    yield chunk

    def foreach_column(self, mapper, reducer):
        self.__check_func(mapper, reducer)

        chunk = self.__chunk_generator()
        data = next(chunk)

        def map_gen(d):
            for col in d.columns:
                yield mapper(d[col])
        
        mapped0 = map_gen(data)

        data = next(chunk, None)
        while (data is not None):
            mapped1 = map_gen(data)
            mapped0 = [
                reducer(m0, m1)
                for m0,m1 in zip(mapped0, mapped1)
            ]
            data = next(chunk, None)
        
        return list(mapped0)
    
    def foreach_chunk(self, mapper, reducer):
        raise Exception('not implemented yet')

    def process_columns(self):
        return self.foreach_column(self.column_mapper, self.column_reducer)
        
    def process_chunks(self):
        return self.foreach_chunk(self.chunk_mapper, self.chunk_reducer)

    