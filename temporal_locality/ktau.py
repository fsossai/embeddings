import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interarrivaltimes import interarrival_times_of
from scipy.stats import kendalltau
from time import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computing correlation between order statistics' +
                    'of embbedding IDs and their average interarrival time'
    )
    parser.add_argument('--feature-index', '-f', type=int, default=14)
    parser.add_argument('--top-orders', '-t', type=int, default=1000)
    args = parser.parse_args()

    inputfile = '..\\data\\day_100k.csv'
    print(f'Reading \'{inputfile}\'')
    data = pd.read_csv(inputfile, sep='\t', header=None)
    vcounts = data[args.feature_index].value_counts()
    vcounts = vcounts[vcounts > 1]
    n_orders = min(vcounts.size, args.top_orders)
    orders = list(range(n_orders))

    print(f'Found {vcounts.size} embedding IDs in feature {args.feature_index}.')

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
        _, inter = interarrival_times_of(id, data, args.feature_index)
        avgit.append(np.mean(inter))
        id_names[j] = id
    print()

    ktau = kendalltau(orders, avgit)
    t = time() - t
    print(f'Elapsed time\t: {t:.5} sec')

    print(f'kendalltau\t: {ktau.correlation}')

    # saving to file
    outname = f'ktau_f{args.feature_index}_' + str(int(time()))
    np.savez(outname, avgit=avgit, ktau=ktau.correlation, top=n_orders)

    # plotting
    plt.plot(orders, avgit, lw=1, label=f'KendallTau = {ktau.correlation:.5}')
    # plt.xticks(orders, id_names, rotation=90)
    plt.xlabel('Order Statistic (OS)')
    plt.ylabel('Average Interarrival time (AvgIT)')
    plt.title(f'Correlation between OS and AvgIT, Feature {args.feature_index}')
    plt.tight_layout()
    plt.savefig(outname + '.png')
    plt.legend()
    plt.show()
