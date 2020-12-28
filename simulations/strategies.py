import numpy as np
import matplotlib.pyplot as plt
from time import time
import simtools
import sys
sys.path.append('..')
import bigdatatools
from bigdatatools import ChunkStreaming

if __name__ == '__main__':
    total_time = time()
    parser = bigdatatools.default_parser()
    parser.add_argument('--queries', '-q', type=str, required=True)
    parser.add_argument('--embedding-tables', '-t', type=str, required=True)
    parser.add_argument('--strategies', '-s',
                        type=simtools.comma_separated_strategies, required=True)
    parser.add_argument('--ndevices', '-d', type=simtools.range_list, required=True)
    args = parser.parse_args()

    strategies = args.strategies.split(',')
    ndevices = bigdatatools.get_range_list(args.ndevices)

    if args.selected_columns is not None:
        selected_columns = bigdatatools.get_range_list(args.selected_columns)
    else:
        selected_columns = None

    pandas_kwargs = {
        'sep': '\t',
        'header': None,
        'usecols': selected_columns,
        'chunksize': args.chunk_size,
        'compression': 'gzip' if args.gzip else None
    }

    for strategy in strategies:
        print(f'Strategy\t: {strategy}')

        # creating a streaming chunk generator
        reader = ChunkStreaming(
            files=args.queries,
            nchunks=args.n_chunks,
            **pandas_kwargs
        ).chunk_gen()

        sim = simtools.Simulation()
        sim.reload_embtables(args.embedding_tables)
        res = sim.run(reader, strategy, ndevices)

        print(res)

    # plotting
