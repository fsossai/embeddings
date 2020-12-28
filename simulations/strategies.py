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
    parser.add_argument('--queries', '-q', type=str, required=True)
    parser.add_argument('--embedding-tables', '-t', type=str, required=True)
    parser.add_argument('--strategies', '-s',
                        type=simtools.comma_separated_strategies, required=True)
    parser.add_argument('--ndevices', '-d', type=bigdatatools.range_list, required=True)
    parser.add_argument('--output-dir', '-D', type=str, default=None)
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

    all_results = dict()
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
        all_results[strategy] = res

    # dumping to file
    output_dir = args.output_dir or '.'
    timestamp = str(int(time()))
    outname = f'{output_dir}\\sim{timestamp}_results.bin'
    with open(outname, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Results saved to \'{outname}\'.')

    # creating a grid of grouped-bar plots
    fig, axs = plt.subplots(1, len(ndevices))
    fig.set_size_inches(1366 / fig.dpi, 681 / fig.dpi)
    fig.suptitle(f'Memory Loads Simulation, queries from \'{args.queries}\'')
    seaborn.set_theme(style="whitegrid")
    # creating a dictionary of device-centered DataFrame
    memloads = dict()
    for i, dev in enumerate(ndevices):
        memloads[dev] = pd.DataFrame({
            'Device': [f'M{i}' for i in range(dev)]
        })
        for strategy in strategies:
            memloads[dev][strategy] = all_results[strategy][dev]['memload']
        # rearranging for ease of plotting with seaborn
        memloads[dev] = memloads[dev].melt(
            id_vars='Device',
            value_name='Memory Load',
            var_name='Strategy'
        )
        seaborn.barplot(
            x='Device',
            y='Memory Load',
            hue='Strategy',
            data=memloads[dev],
            ax=axs[i]
        )
        axs[i].set(ylabel='', title=f'{dev} memory devices')

    # saving to file memload
    axs[0].set_ylabel('Memory Load (queries per device')
    plt.tight_layout()
    fig.savefig(f'{output_dir}\\sim{timestamp}_memload.png')
    fig.savefig(f'{output_dir}\\sim{timestamp}_memload.svg')

    # creating final average fanout plot
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Fanout Simulation, queries from \'{args.queries}\'')
    # plotting theoretical random fanout
    random_fanout = simtools.random_strategy_fanout(ndevices, len(selected_columns))
    ax.plot(ndevices, random_fanout, '--', label='random')
    # plotting fanout of the other strategies
    for strategy in strategies:
        avgfanout = [
            all_results[strategy][dev]['avgfanout']
            for dev in ndevices
        ]
        ax.plot(ndevices, avgfanout, label=strategy)

    ax.set(xlabel='Number of memory devices', ylabel='Average fanout')
    fig.tight_layout()
    ax.legend()
    fig.savefig(f'{output_dir}\\sim{timestamp}_fanout.png')
    fig.savefig(f'{output_dir}\\sim{timestamp}_fanout.svg')

    plt.show()










