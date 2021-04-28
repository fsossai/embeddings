import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import seaborn
import json
import sys
import os

reordering = [19, 0, 21, 9, 20, 10, 11, 22, 4, 1, 2, 23, 6, 3, 14, 13, 7, 17, 15, 24, 8, 25, 18, 12, 16, 5]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json','-i', type=str, required=True)
    parser.add_argument('--save','-s', action="store_true", default=False)
    parser.add_argument('--which-plots','-p', type=str, default='RP,OP,OL,PS')
    parser.add_argument('--reordering','-r', action="store_true", default=False)
    args = parser.parse_args()

    if args.which_plots.lower() == 'all':
        args.which_plots = 'PM,LM,RP,LR,F,OP,OL,PS,OT,C,CT,CP,FP'

    # checking input arguments
    if not os.path.isfile(args.input_json):
        print('ERROR: file not found')
        sys.exit(-1)

    # loading JSON simulation data
    with open(args.input_json, 'r') as f:
        sim = json.load(f)

    # getting the name of the simulation
    name = os.path.splitext(args.input_json)[0]

    P, D, N = sim['processors'], sim['tables'], sim['queries']
    
    # loading matrices and arrays from the JSON dictionary
    packets = np.array(sim['packets'])
    lookups = np.array(sim['lookups'])
    outgoing_packets = np.array(sim['outgoing_packets'])
    outgoing_lookups = np.array(sim['outgoing_lookups'])
    outgoing_tables = np.array(sim['outgoing_tables'])
    packet_size = np.array(sim['packet_size'])
    fanout = np.array(sim['fanout'])
    cache_hits = None
    df_footprint = None
    if 'cache_hits' in sim:
        cache_hits = np.array(sim['cache_hits'])
        cache_refs = np.array(sim['cache_refs'])
        cache_sizes = np.array(sim['cache_sizes'])
        if 'cache_footprint' in sim:
            cache_footprint = sim['cache_footprint']
            df_footprint = pd.DataFrame(cache_footprint)
            df_footprint = df_footprint.reindex(index=df_footprint.index[::-1])
            

    # calculating averages
    avg = lambda x: (x * np.arange(len(x))).sum() / x.sum()
    avg_fanout = avg(fanout)
    avg_out_packets = avg(outgoing_packets)
    avg_out_lookups = avg(outgoing_lookups)
    avg_packet_size = np.array([avg(packet_size[i, :]) for i in range(P)])
    # checking consistency of averages
    assert avg_out_packets == packets.sum() / N, "avg_out_packets corrupted"
    assert avg_out_lookups == lookups.sum() / N, "avg_out_lookups corrupted"
    assert avg(packet_size.sum(axis=0)) == lookups.sum() / packets.sum(), "avg_packet_size corrupted"

    # calculating load imbalance
    dispersion = lambda x: x.std() / x.mean()
    packets_imb = dispersion(packets.sum(axis=0))
    lookups_imb = dispersion(lookups.sum(axis=0))

    # plotting results

    if 'PM' in args.which_plots.split(','):
        plt.figure(0)
        plt.title('Packets matrix')
        plt.imshow(packets)
        plt.colorbar()
        plt.xticks(range(P), range(P))
        plt.yticks(range(P), range(P))
        plt.tight_layout()
        if args.save:
            plt.savefig(name + '_PM.png')

    if 'LM' in args.which_plots.split(','):
        plt.figure(1)
        plt.title('Lookups matrix')
        plt.imshow(lookups)
        plt.colorbar()
        plt.xticks(range(P), range(P))
        plt.yticks(range(P), range(P))
        plt.tight_layout()
        if args.save:
            plt.savefig(name + '_LM.png')

    if 'RP' in args.which_plots.split(','):
        plt.figure(2)
        plt.title(f'Received packets, imbalance={packets_imb:.3}')
        plt.bar(range(P), packets.sum(axis=0))
        plt.xticks(range(P), range(P))
        plt.xlabel('Processor')
        plt.ylabel('Number of received packets')
        plt.tight_layout()
        if args.save:
            plt.savefig(name + '_RP.png')
            
    if 'LR' in args.which_plots.split(','):
        plt.figure(3)
        plt.title(f'Lookup requests, imbalance={lookups_imb:.3}')
        plt.bar(range(P), lookups.sum(axis=0))
        plt.xticks(range(P), range(P))
        plt.xlabel('Processor')
        plt.ylabel('Number of lookup requests')
        plt.tight_layout()
        if args.save:
            plt.savefig(name + '_LR.png')

    if 'F' in args.which_plots.split(','):
        plt.figure(4)
        plt.bar(range(1,P+1), fanout[1:])
        plt.title(f'Fanout distribution, avg={avg_fanout:.3}')
        plt.xlabel('Fanout')
        plt.xticks(range(1,P+1), range(1,P+1), rotation=0)
        plt.ylabel('Count')
        plt.tight_layout()
        if args.save:
            plt.savefig(name + '_F.png')

    if 'OP' in args.which_plots.split(','):
        plt.figure(5)
        plt.bar(range(P), outgoing_packets)
        plt.title(f'Outgoing packets distribution, avg={avg_out_packets:.3}')
        plt.xlabel('Number of outgoing packets')
        plt.xticks(range(P), range(P))
        plt.ylabel('Count')
        plt.tight_layout()
        if args.save:
            plt.savefig(name + '_OP.png')

    if 'OL' in args.which_plots.split(','):
        plt.figure(6)
        plt.bar(range(D+1), outgoing_lookups)
        plt.title(f'Outgoing lookups distribution, avg={avg_out_lookups:.3}')
        plt.xlabel('Number of outgoing packets')
        plt.xticks(range(D+1), range(D+1), rotation=90)
        plt.ylabel('Count')
        plt.tight_layout()
        if args.save:
            plt.savefig(name + '_OL.png')

    if 'OT' in args.which_plots.split(','):
        plt.figure(7)
        plt.title('Outgoing tables matrix')
        plt.xlabel('Table index')
        plt.ylabel('Processor')
        if args.reordering:
            plt.imshow(outgoing_tables[:, reordering])
            plt.xticks(range(D), reordering, rotation=90)
        else:
            plt.xticks(range(D), range(D), rotation=90)
            plt.imshow(outgoing_tables)
        plt.colorbar()
        plt.yticks(range(P), range(P))
        plt.tight_layout()
        if args.save:
            plt.savefig(name + '_OT.png')
    
    if 'PS' in args.which_plots.split(','):
        plt.figure(8)
        plt.bar(range(1,D+1), packet_size.sum(axis=0)[1:])
        plt.title(f'Packet size distribution, avg={avg(packet_size.sum(axis=0)):.3}')
        plt.xlabel('Packet size')
        plt.xticks(range(1,D+1), range(1,D+1), rotation=90)
        plt.ylabel('Count')
        '''
        plt.figure(8)
        plt.bar(range(P), avg_packet_size)
        plt.title(f'Average packet size, total avg={avg(packet_size.sum(axis=0)):.3}')
        plt.xlabel('Processor')
        plt.xticks(range(P), range(P), rotation=0)
        plt.ylabel('Packet size')
        '''
        plt.tight_layout()
        if args.save:
            plt.savefig(name + '_PS.png')

    if cache_hits is not None:
        if 'C' in args.which_plots.split(','):
            plt.figure(9)
            plt.title('Cache hit-rates')
            if args.reordering:
                plt.imshow((cache_hits / cache_refs)[:, reordering])
                plt.xticks(range(D), reordering, rotation=90)
            else:
                plt.imshow(cache_hits / cache_refs)
                plt.xticks(range(D), range(D), rotation=90)
            plt.colorbar()
            plt.xlabel('Table index')
            plt.ylabel('Processor index')
            plt.yticks(range(P), range(P))
            plt.tight_layout()
            if args.save:
                plt.savefig(name + '_C.png')

        if 'CT' in args.which_plots.split(','):
            plt.figure(10)
            plt.title('Cache hit-rates by table')
            if args.reordering:
                plt.bar(range(D), (cache_hits.sum(axis=0) / cache_refs.sum(axis=0))[reordering])
                plt.xticks(range(D), reordering, rotation=90)
            else:
                plt.bar(range(D), cache_hits.sum(axis=0) / cache_refs.sum(axis=0))
                plt.xticks(range(D), range(D), rotation=90)
            plt.xlabel('Table index')
            plt.ylabel('Hit-rate')
            plt.tight_layout()
            if args.save:
                plt.savefig(name + '_CT.png')

        if 'CP' in args.which_plots.split(','):
            plt.figure(11)
            plt.title('Cache hit-rates by processor')
            plt.bar(range(P), cache_hits.sum(axis=1) / cache_refs.sum(axis=1))
            plt.xlabel('Processor index')
            plt.ylabel('Hit-rate')
            plt.xticks(range(P), range(P))
            plt.tight_layout()
            if args.save:
                plt.savefig(name + '_CP.png')

    if df_footprint is not None:
        if 'FP' in args.which_plots.split(','):
            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [P, 1]})
            if args.reordering:
                df_footprint[reordering].plot.barh(stacked=True, legend=False, ax=axs[0])
            else:
                df_footprint.plot.barh(stacked=True, legend=False, ax=axs[0])
            axs[0].set(ylabel='Processor index')
            axs[0].set(title="Tables' cache footprint")
            axs[0].set_xticks([])
            axs[0].set_xticklabels([])
            if args.reordering:
                pd.DataFrame([cache_sizes])[reordering].plot.barh(stacked=True, legend=False, ax=axs[1])
            else:
                pd.DataFrame([cache_sizes]).plot.barh(stacked=True, legend=False, ax=axs[1])
            axs[1].set(xlabel="Reference footprint")
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            axs[1].set_xticklabels([])
            axs[1].set_yticklabels([])
            fig.tight_layout()
            if args.save:
                plt.savefig(name + '_FP.png')

    if not args.save:
        plt.show()
