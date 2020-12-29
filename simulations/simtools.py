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
    avail_strat = available_strategies()
    for s in value.split(','):
        if s not in avail_strat:
            raise argparse.ArgumentTypeError(
                f'\'{s}\' is not an available strategy')
    return value


def create_tables_from_dataframe(inputfiles, outputfile, nchunks,
                                 selected_columns=None, **pandas_kwargs):
    cs = ChunkStreaming(inputfiles, nchunks=nchunks, **pandas_kwargs)
    cs.column_mapper = pd.Series.value_counts
    cs.column_feeder = selected_columns
    cs.column_reducer = lambda x, y: pd.concat([x, y]).groupby(level=0).sum()

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


def available_strategies():
    return [
        'random',
        'greedy',
        'roundrobin1',
        'roundrobin2'
    ]


def available_allocations():
    return [
        'fit'
    ]


def random_strategy_fanout(ndevices, features):
    return [
        n * (1 - np.power(1 - 1 / n, features))
        for n in ndevices
    ]


class Simulation:
    def __init__(self):
        self.selected_columns = None
        self.tables = None
        self.data_available = False
        self.processed_queries = 0

    def load_tables_from_dataframe(self, inputfiles, selected_columns=None, **pandas_kwargs):
        # processing a pandas.DataFrame with ChunkStreaming
        self.selected_columns = selected_columns  # (?)
        cs = ChunkStreaming(
            files=inputfiles,
            drop_nan=False,
            nchunks=np.inf,
            **pandas_kwargs
        )
        cs.column_mapper = pd.Series.value_counts
        cs.column_feeder = selected_columns
        cs.column_reducer = lambda x, y: pd.concat([x, y]).groupby(level=0).sum()

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
        print(f'Sharding\t: ', end='', flush=True)
        for dev in ndevices:
            print(f'{dev} ', end='', flush=True)
            m = MemorySystem(dev)
            m.load_embtables(self.tables)
            m.shard_embeddings(
                strategy=strategy,
                allocation='fit'  # for now it's the only one implemented
            )
            m.set_active(0)
            mems[dev] = m
        print()

        for i, chunk in enumerate(reader):
            print(f'Devices\t: ', end='', flush=True)
            for dev in ndevices:
                print(f'{dev} ', end='', flush=True)
                mems[dev].lookups(index=0, queries=chunk)
            self.processed_queries += len(chunk)
        print()

        avgfanout = [(dev, np.mean(mems[dev].fanout)) for dev in ndevices]
        memload = [(dev, mems[dev].requests_per_dev) for dev in ndevices]
        return dict({
            'avgfanout': avgfanout,
            'memload': memload
        })


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
            self.tables = {k: tables[k].copy() for k in tables}
        else:
            # checking tables type
            if any(type(y) is not pd.Series for x, y in tables):
                raise Exception(f'wrong embedding table format')
            self.tables = {x: y.copy() for x, y in tables}

    def shard_embeddings(self, **configuration):
        # checking missing parameters
        if 'strategy' not in configuration or \
                'allocation' not in configuration:
            raise Exception('missing some configuration parameters')

        # checking arguments
        if configuration['strategy'] not in available_strategies():
            raise Exception('invalid strategy')
        if configuration['allocation'] not in available_allocations():
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
        dcm_gen = self.device_cm_gen(size_single)
        for t in self.tables:
            devs = [next(dcm_gen) for i in range(self.tables[t].size)]
            self.tables[t][:] = devs

    # tables row-major / devices column-major
    def shard_roundrobin1(self):
        self.check_sharding_requirements()
        size_single = self.min_size_single()

        # sharding
        dcm_gen = self.device_cm_gen(size_single)
        trm_gen = self.tables_rm_gen()
        for t, i in trm_gen:
            self.tables[t][i] = next(dcm_gen)

    # tables column-major / devices row-major
    def shard_roundrobin2(self):
        self.check_sharding_requirements()
        size_single = self.min_size_single()

        # sharding
        drm_gen = self.device_rm_gen(size_single)
        for t in self.tables:
            self.tables[t][:] = [
                next(drm_gen)
                for i in range(self.tables[t].size)
            ]

    def min_size_single(self):
        n_ids = sum(self.tables[t].size for t in self.tables)
        reminder = n_ids % self.ndevices
        return n_ids // self.ndevices + min(reminder, 1)

    # device column-major generator
    def device_cm_gen(self, size_single):
        for d in range(self.ndevices):
            for i in range(size_single):
                yield d

    # device row-major generator
    def device_rm_gen(self, size_single):
        for i in range(size_single):
            for d in range(self.ndevices):
                yield d

    def tables_rm_gen(self):
        tuple_table = (
            [
                (t, i)
                for i in range(self.tables[t].size)
            ]
            for t in self.tables
        )
        for row in zip_discard(*tuple_table):
            for col in row:
                yield col

    def lookup(self, index, query):
        if not self.is_active(index):
            raise Exception('inactive devices cannot lookup for queries')
        # lookup for query
        lkup = []
        for t, id in query.iteritems():
            dev_index = self.tables[t][id]
            self.requests_per_dev[dev_index] += 1
            lkup.append(dev_index)
        return lkup

    def lookups(self, index, queries):
        for _, q in queries.iterrows():
            lkup = self.lookup(index, q)
            self.fanout.append(len(set(lkup)))
