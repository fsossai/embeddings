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

    # plotting results

    plt.figure(0)
    plt.title('Packets matrix')
    plt.imshow(packets)
    plt.colorbar()
    plt.xticks(range(P), range(P))
    plt.yticks(range(P), range(P))

    plt.figure(1)
    plt.title('Lookups matrix')
    plt.imshow(lookups)
    plt.colorbar()
    plt.xticks(range(P), range(P))
    plt.yticks(range(P), range(P))

    plt.figure(2)
    plt.title('Received packets')
    plt.bar(range(P), packets.sum(axis=0))
    plt.xticks(range(P), range(P))
    plt.xlabel('Processor')
    plt.ylabel('Number of received packets')
    plt.tight_layout()

    plt.figure(3)
    plt.title('Lookup requests')
    plt.bar(range(P), lookups.sum(axis=0))
    plt.xticks(range(P), range(P))
    plt.xlabel('Processor')
    plt.ylabel('Number of lookup requests')
    plt.tight_layout()

    plt.figure(4)
    plt.bar(range(0,P+1), fanout)
    plt.title('Fanout distribution')
    plt.xlabel('Fanout')
    plt.xticks(range(0,P+1), range(0,P+1), rotation=0)
    plt.yticks(None, None)
    plt.ylabel('Count')
    plt.tight_layout()

    plt.figure(5)
    plt.bar(range(0,D+1), outgoing_packets)
    plt.title('Outgoing packets distribution')
    plt.xlabel('Number of outgoing packets')
    plt.xticks(range(0,D+1), range(0,D+1), rotation=90)
    plt.yticks(None, None)
    plt.ylabel('Count')
    plt.tight_layout()

    plt.figure(6)
    plt.bar(range(0,D+1), outgoing_lookups)
    plt.title('Outgoing lookups distribution')
    plt.xlabel('Number of outgoing lookups')
    plt.xticks(range(0,D+1), range(0,D+1), rotation=90)
    plt.yticks(None, None)
    plt.ylabel('Count')
    plt.tight_layout()

    plt.show()