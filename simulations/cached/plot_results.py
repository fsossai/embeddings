import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import argparse
from glob import glob
import sys
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json','-i', type=str, required=True)
    args = parser.parse_args()

    # checking input arguments
    if len(glob(args.input_json)) == 0:
        print('ERROR: file not found')
        sys.exit(-1)

    with open(args.input_json, 'r') as f:
        sim = json.load(f)

    P, D, N = sim['processors'], sim['tables'], sim['queries']
    
    # loading matrices and arrays from the JSON dictionary
    packets = np.array(sim['packets'])
    lookups = np.array(sim['lookups'])
    outgoing_packets = np.array(sim['outgoing_packets'])
    outgoing_lookups = np.array(sim['outgoing_lookups'])
    fanout = np.array(sim['fanout'])
    cache_hits = None
    if 'cache_hits' in sim:
        cache_hits = np.array(sim['cache_hits'])
        cache_refs = np.array(sim['cache_refs'])

    # calculating averages
    avg = lambda x: (x * np.arange(0, len(x))).sum() / x.sum()
    avg_fanout = avg(fanout)
    avg_out_packets = avg(outgoing_packets)
    avg_out_lookups = avg(outgoing_lookups)

    # plotting results

    plt.figure(0)
    plt.title('Packets matrix')
    plt.imshow(packets)
    plt.colorbar()
    plt.xticks(range(P), range(P))
    plt.yticks(range(P), range(P))
    plt.tight_layout()

    plt.figure(1)
    plt.title('Lookups matrix')
    plt.imshow(lookups)
    plt.colorbar()
    plt.xticks(range(P), range(P))
    plt.yticks(range(P), range(P))
    plt.tight_layout()

    plt.figure(2)
    plt.title(f'Received packets')
    plt.bar(range(P), packets.sum(axis=0))
    plt.xticks(range(P), range(P))
    plt.xlabel('Processor')
    plt.ylabel('Number of received packets')
    plt.tight_layout()

    plt.figure(3)
    plt.title(f'Lookup requests')
    plt.bar(range(P), lookups.sum(axis=0))
    plt.xticks(range(P), range(P))
    plt.xlabel('Processor')
    plt.ylabel('Number of lookup requests')
    plt.tight_layout()

    plt.figure(4)
    plt.bar(range(0,P+1), fanout)
    plt.title(f'Fanout distribution, avg={avg_fanout:.3}')
    plt.xlabel('Fanout')
    plt.xticks(range(0,P+1), range(0,P+1), rotation=0)
    plt.yticks(None, None)
    plt.ylabel('Count')
    plt.tight_layout()

    plt.figure(5)
    plt.bar(range(0,P+1), outgoing_packets)
    plt.title(f'Outgoing packets distribution, avg={avg_out_packets:.3}')
    plt.xlabel('Number of outgoing packets')
    plt.xticks(range(0,P+1), range(0,P+1))
    plt.yticks(None, None)
    plt.ylabel('Count')
    plt.tight_layout()

    plt.figure(6)
    plt.bar(range(0,D+1), outgoing_lookups)
    plt.title(f'Outgoing lookups distribution, avg={avg_out_lookups:.3}')
    plt.xlabel('Number of outgoing lookups')
    plt.xticks(range(0,D+1), range(0,D+1), rotation=90)
    plt.yticks(None, None)
    plt.ylabel('Count')
    plt.tight_layout()

    if cache_hits is not None:
        plt.figure(7)
        plt.title('Cache hit-rates')
        plt.imshow(cache_hits / cache_refs)
        plt.colorbar()
        plt.xlabel('Table index')
        plt.ylabel('Processor index')
        plt.xticks(range(D), range(D), rotation=90)
        plt.yticks(range(P), range(P))
        plt.tight_layout()


    plt.show()