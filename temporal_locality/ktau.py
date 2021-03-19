import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interarrivaltimes import interarrival_times_of
from scipy.stats import kendalltau
from time import time
import argparse
import sys; sys.path.append('..')
import bigdatatools
from bigdatatools import ChunkStreaming

if __name__ == '__main__':
    parser = bigdatatools.default_parser()
    parser.description = 'Computing correlation between order statistics ' \
                         'of embbedding IDs and their average interarrival time.'
    parser.add_argument('--files', '-f', type=str, default=None, required=True)
    #parser.add_argument('--feature-index', '-f', type=int, default=14)
    parser.add_argument('--top-orders', '-t', type=int, default=1000)
    #parser.add_argument('--file', '-f', type=str, default=None, required=True)
    args = parser.parse_args()

    column_selection = bigdatatools.get_range_list(args.column_selection)
    inputfile = args.files
    print(f'Reading \'{inputfile}\'')
    data = pd.read_csv(inputfile, sep='\t', header=None)
    vcounts = data[column_selection[0]].value_counts()
    vcounts = vcounts[vcounts > 1]
    n_orders = min(vcounts.size, args.top_orders)
    orders = list(range(n_orders))

    print(f'Found {vcounts.size} embedding IDs in feature {column_selection[0]}.')

    t = time()
    avgit = []
    id_names = [None] * n_orders
    print_step = 10
    for j, (id, counts) in enumerate(vcounts.head(n_orders).iteritems()):
        if j % print_step == 0:
            print(f'\r Computing average interarrival times of \'{id}\'\t' +
                  f'{(j + 1) / n_orders * 100:.4}%          ',
                  end='', flush=True
                  )
        _, inter = interarrival_times_of(id, data, column_selection[0])
        avgit.append(np.mean(inter))
        id_names[j] = id
    print()

    ktau = kendalltau(orders, avgit)
    t = time() - t
    print(f'Elapsed time\t: {t:.5} sec')

    print(f'kendalltau\t: {ktau.correlation}')

    # saving to file
    outname = f'ktau_f{column_selection[0]}_' + str(int(time()))
    np.savez(outname, avgit=avgit, ktau=ktau.correlation, top=n_orders)

    # plotting
    plt.plot(orders, avgit, lw=1, label=f'KendallTau = {ktau.correlation:.5}')
    # plt.xticks(orders, id_names, rotation=90)
    plt.xlabel('Order Statistic (OS)')
    plt.ylabel('Average Interarrival time (AvgIT)')
    plt.title(f'Correlation between OS and AvgIT, Feature {column_selection[0]}')
    plt.tight_layout()
    plt.savefig(outname + '.png')
    plt.legend()
    plt.show()
