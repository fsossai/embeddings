import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import argparse
from itertools import zip_longest
import pickle
import sys
sys.path.append('..')
from bigdatatools import ChunkStreaming

def zip_discard(*iterables):
    return map(
            lambda x: filter(lambda y: y is not None, x),
            zip_longest(*iterables)
    )


def comma_separated_strategies(value):
    all = MemorySystem.available_strategies()
    for s in value.split(','):
        if s not in all:
            raise argparse.ArgumentTypeError(
                f'\'{s}\' is not an available strategy')
    return value


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


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-size','-c', type=int, default=int(1e6))
    parser.add_argument('--n-chunks','-n', type=int, default=None)
    parser.add_argument('--gzip','-z', action='store_true', default=False)
    parser.add_argument('--selected-columns', '-S', type=range_list, default=None)
    return parser


def create_tables_from_dataframe(inputfiles, outputfile, nchunks,
    selected_columns=None, **pandas_kwargs):

    cs = ChunkStreaming(inputfiles, nchunks=nchunks, **pandas_kwargs)
    cs.column_mapper = pd.Series.value_counts
    cs.column_feeder = selected_columns
    cs.column_reducer = lambda x,y: pd.concat([x, y]).groupby(level=0).sum()
    
    # computing
    print(f'Embedding table creation started.')
    t = time()
    vcounts = dict(cs.process_columns())
    t = time() - t
    print(f'Elapsed time\t: {t:.5} sec')

    # saving to file
    t = time()
    print(f'Saving to \'{outputfile}\' ... ', end='', flush=True)
    with open(outputfile, 'wb') as f:
        pickle.dump(vcounts, f)
    t = time() - t
    print(f'{t:.5} sec')


class Simulation:
    def __init__(self):
        self.selected_columns = None
        self.tables = None
        self.data_available = False
    

    def load_tables_from_dataframe(self, inputfiles, selected_columns=None, **pandas_kwargs):
        # processing a pandas.DataFrame with ChunkStreaming
        self.selected_columns = selected_columns # (?)
        cs = ChunkStreaming(
            files=inputfiles,
            drop_nan=False,
            nchunks=args.n_chunks,
            **pandas_kwargs
        )
        cs.column_mapper = pd.Series.value_counts
        cs.column_feeder = selected_columns
        cs.column_reducer = lambda x,y: pd.concat([x, y]).groupby(level=0).sum()

        print(f'Loading queries.')
        t = time()
        self.tables = dict(cs.process_columns())
        t = time() - t
        print(f'Elapsed time\t: {t:.5} sec')
        self.data_available = True


    def reload_embtables(self, file):
        # open already processed file
        with open(file, 'rb') as f:
            self.tables = pickle.load(f)
        self.data_available = True


    def save_embtables(self, file):
        # saving tables to file
        with open(file, 'wb') as f:
            pickle.dump(self.tables, f)


    def run(self, reader, strategy, ndevices):
        if type(ndevices) is not list:
            ndevices = [ndevices]

        mems = dict()
        for dev in ndevices:
            m = MemorySystem(dev)
            m.load_embtables(self.tables)
            m.shard_embeddings(
                strategy = strategy,
                allocation = 'fit' # for now it's the only one implemented
            )
            m.set_active(0)
            mems[dev] = m

        for i,chunk in enumerate(reader):
            print(f'Devices\t: ', end='', flush=True)
            for dev in ndevices:
                print(f'{dev} ', end='', flush=True)
                mems[dev].lookups(index=0, queries=chunk)
        print()
        
        results = dict()
        for dev in ndevices:
            res = {
                'avgfanout' : np.mean(mems[dev].fanout),
                'memload' : mems[dev].requests_per_dev
            }
            results[dev] = res
        return results


class MemorySystem:
    def __init__(self, ndevices=1):
        self.ndevices = ndevices
        self.__active = [False] * ndevices
        self.requests_per_dev = [0] * ndevices
        self.fanout = []
        self.tables = None
        self.configuration = None


    def set_active(self, index):
        self.__active[index] = True


    def is_active(self, index):
        return self.__active[index]


    def load_embtables(self, tables):
        # performs a deep copy of 'tables'
        if type(tables) is dict:
            # checking tables type
            if any(type(x) is not pd.Series for x in tables.values()):
                raise Exception(f'wrong embedding table format')
            self.tables = { k:tables[k].copy() for k in tables }
        else:
            # checking tables type
            if any(type(y) is not pd.Series for x,y in tables):
                raise Exception(f'wrong embedding table format')
            self.tables = { x:y.copy() for x,y in tables }


    def shard_embeddings(self, **configuration):
        # checking missing parameters
        if  'strategy' not in configuration or \
            'allocation' not in configuration:
            raise Exception('missing some configuration parameters')
        
        # checking arguments
        if configuration['strategy'] not in MemorySystem.available_strategies():
            raise Exception('invalid strategy')
        if configuration['allocation'] not in MemorySystem.available_allocations():
            raise Exception('invalid allocation')
        self.configuration = configuration
        
        # sharding, finally
        if configuration['strategy'] == 'random':
            self.shard_random()
        elif configuration['strategy'] == 'greedy':
            self.shard_greedy()
        elif configuration['strategy'] == 'roundrobin1':
            self.shard_roundrobin1()
        elif configuration['strategy'] == 'roundrobin2':
            self.shard_roundrobin2()
        else:
            raise Exception('strategy not implemented')


    def check_sharding_requirements(self):
        if self.configuration is None:
            raise Exception('missing configuration')
        if self.tables is None:
            raise Exception('missing embedding tables')


    def shard_random(self):
        self.check_sharding_requirements()
        for t in self.tables:
            self.tables[t][:] = np.random.randint(
                0, self.ndevices, self.tables[t].size)


    def shard_greedy(self):
        self.check_sharding_requirements()
        # computing the necessary size of a single memory
        size_single = self.min_size_single()
        # sharding
        dcm_gen = self.device_CM_gen(size_single)
        for t in self.tables:
            devs = [next(dcm_gen) for i in range(self.tables[t].size)]
            self.tables[t][:] = devs


    # tables row-major / devices column-major
    def shard_roundrobin1(self):
        self.check_sharding_requirements()
        size_single = self.min_size_single()

        # sharding
        dcm_gen = self.device_CM_gen(size_single)
        trm_gen = self.tables_RM_gen()
        for t,i in trm_gen:
            self.tables[t][i] = next(dcm_gen)


    # tables column-major / devices row-major
    def shard_roundrobin2(self):
        self.check_sharding_requirements()
        size_single = self.min_size_single()

        # sharding
        drm_gen = self.device_RM_gen(size_single)
        for t in self.tables:
            self.tables[t][:] = [
                next(drm_gen)
                for i in range(self.tables[t].size)
            ]
    

    def min_size_single(self):
        n_ids = sum(self.tables[t].size for t in self.tables)
        reminder = n_ids % self.ndevices
        return n_ids // self.ndevices + min(reminder,1)


    # device column-major generator
    def device_CM_gen(self, size_single):
        for d in range(self.ndevices):
            for i in range(size_single):
                yield d


    # device row-major generator
    def device_RM_gen(self, size_single):
        for i in range(size_single):
            for d in range(self.ndevices):
                yield d


    def tables_RM_gen(self):
        tuple_table = (
            (
                (t,i)
                for i in range(self.tables[t].size)
            )
            for t in self.tables
        )
        for row in zip_discard(*tuple_table):
            for col in row:
                yield col


    def available_strategies():
        return [
            'random',
            'greedy',
            'roundrobin'
        ]


    def available_allocations():
        return [
            'fit'
        ]


    def lookup(self, index, query):
        if not self.is_active(index):
            raise Exception('inactive devices cannot lookup for queries')
        # lookup for query
        lkup = []
        for t,id in query.iteritems():
            l = self.tables[t][id]
            #print('l',l,'/',self.ndevices)
            try:
                self.requests_per_dev[l] += 1
            except:
                print(self.tables[t].unique())
                raise Exception()
            lkup.append(l)
        return lkup
        

    def lookups(self, index, queries):
        for _,q in queries.iterrows():
            lkup = self.lookup(index, q)
            self.fanout.append(len(set(lkup)))




