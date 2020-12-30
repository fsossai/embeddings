import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from time import time
import pickle
import simtools
import sys
sys.path.append('..')
import bigdatatools
from bigdatatools import ChunkStreaming

if __name__ == '__main__':
    total_time = time()
    parser = bigdatatools.default_parser()
    parser.add_argument('--queries-file', '-q', type=str, required=True)
    parser.add_argument('--embedding-tables', '-t', type=str, required=True)
    parser.add_argument('--strategies', '-s',
                        type=simtools.comma_separated_strategies, required=True)
    parser.add_argument('--n-devices', '-d', type=bigdatatools.range_list, required=True)
    parser.add_argument('--output-dir', '-D', type=str, default=None)
    args = parser.parse_args()

    strategies = args.strategies.split(',')
    ndevices = bigdatatools.get_range_list(args.n_devices)

    if args.column_selection is not None:
        column_selection = bigdatatools.get_range_list(args.column_selection)
    else:
        column_selection = None

    pandas_kwargs = {
        'sep': '\t',
        'header': None,
        'usecols': column_selection,
        'chunksize': args.chunk_size,
        'compression': 'gzip' if args.gzip else None
    }

    total_queries = 0
    # creating a dictionary of device-centered DataFrame
    memload = dict()
    for dev in ndevices:
        memload[dev] = pd.DataFrame({
            'Device': [f'M{i}' for i in range(dev)]
        })


    avgfanout = dict()
    memload = dict()
    for strategy in strategies:
        print(f'Strategy\t: {strategy}')
        # creating a streaming chunk generator
        reader = ChunkStreaming(
            files=args.queries_file,
            nchunks=args.n_chunks,
            **pandas_kwargs
        ).chunk_gen()

        # running simulation
        sim = simtools.Simulation()
        sim.reload_embtables(args.embedding_tables)
        res = sim.run(reader, strategy, ndevices)

        #processing results
        total_queries = max(total_queries, sim.processed_queries)
        avgfanout[strategy] = res['avgfanout']
        memload[strategy] = res['memload']

    # creating device-centered DataFrames
    memload_dfs = dict()
    for dev in ndevices:
        memload_dfs[dev] = pd.DataFrame({
            'Device': [f'M{i}' for i in range(dev)]
        })
        for strategy in strategies:
            memload_dfs[dev][strategy] = dict(memload[strategy])[dev]
        memload_dfs[dev] = memload_dfs[dev].melt(
            id_vars='Device',
            value_name='Memory Load',
            var_name='Strategy'
        )

    sim_results = {
        'ndevices': ndevices,
        'avgfanout': avgfanout,
        'memload': memload,
        'nqueries': total_queries,
        'embtables': args.embedding_tables,
        'queriesfile': args.queries_file,
        'memload_dataframe': memload_dfs
    }

    # dumping to file
    output_dir = args.output_dir or '.'
    timestamp = str(int(time()))
    outname = f'{output_dir}\\sim{timestamp}_results.bin'
    with open(outname, 'wb') as f:
        pickle.dump(sim_results, f)
    print(f'Results saved to \'{outname}\'.')

    # creating a grid of grouped-bar plots
    fig, axs = plt.subplots(1, len(ndevices))
    fig.set_size_inches(1366 / fig.dpi, 681 / fig.dpi)
    fig.suptitle(f'Memory Loads Simulation, {total_queries} '
                 f'queries from \'{args.queries_file}\'')
    seaborn.set_theme(style="whitegrid")

    # creating the grouped-bar histograms
    for i, dev in enumerate(ndevices):
        seaborn.barplot(
            x='Device',
            y='Memory Load',
            hue='Strategy',
            data=memload_dfs[dev],
            ax=axs[i]
        )
        axs[i].set(ylabel='', title=f'{dev} memory devices')

    # saving memload images to file
    axs[0].set_ylabel('Memory Load (queries per device)')
    plt.tight_layout()
    fig.savefig(f'{output_dir}\\sim{timestamp}_memload.png')
    fig.savefig(f'{output_dir}\\sim{timestamp}_memload.svg')

    # creating final average fanout plot
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Fanout Simulation, {total_queries} '
                 f'queries from \'{args.queries_file}\'')
    # plotting theoretical random fanout
    random_fanout = simtools.random_strategy_fanout(ndevices, len(column_selection))
    ax.plot(ndevices, random_fanout, '.--', label='random (theoretical)')
    # plotting fanout of the other strategies
    for strategy in strategies:
        avgfanout = dict(sim_results['avgfanout'][strategy]).values()
        ax.plot(ndevices, avgfanout, '.-', label=strategy)

    ax.set(xlabel='Number of memory devices', ylabel='Average fanout')
    fig.tight_layout()
    ax.legend()
    fig.savefig(f'{output_dir}\\sim{timestamp}_fanout.png')
    fig.savefig(f'{output_dir}\\sim{timestamp}_fanout.svg')

    plt.show()










