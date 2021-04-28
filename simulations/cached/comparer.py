from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import seaborn
import json
import time
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
    input_files = args.input_json.split(',')
    n_files = len(input_files)
    for file in input_files:
        if not os.path.isfile(file):
            print(f'ERROR: \"{file}\" file not found')
            sys.exit(-1)

    # loading JSON simulation data
    sim = dict()
    for file in input_files:
        with open(file, 'r') as f:
            sim[file] = json.load(f)

    # getting the name of the simulation
    name = f'comparison{int(time.time())}'

    # checking consistency of P, D and N
    P = sim[input_files[0]]['processors']
    D = sim[input_files[0]]['tables']
    N = sim[input_files[0]]['queries']
    for file in input_files:
        if sim[file]['processors'] != P or \
           sim[file]['tables'] != D or \
           sim[file]['queries'] != N:
           printf('ERROR: P, D, N inconsistency')
           sys.exit(-1)
    
    # loading matrices and arrays from all of the JSON dictionary
    packets = dict()
    lookups = dict()
    outgoing_packets = dict()
    outgoing_lookups = dict()
    outgoing_tables  = dict()
    packet_size  = dict()
    fanout = dict()
    cache_hits = dict()
    cache_refs = dict()
    cache_sizes = dict()
    df_footprint = dict()
    cache_footprint = dict()
    name = dict()

    for file in input_files:
        packets[file] = np.array(sim[file]['packets'])
        lookups[file] = np.array(sim[file]['lookups'])
        outgoing_packets[file] = np.array(sim[file]['outgoing_packets'])
        outgoing_lookups[file] = np.array(sim[file]['outgoing_lookups'])
        outgoing_tables[file] = np.array(sim[file]['outgoing_tables'])
        packet_size[file] = np.array(sim[file]['packet_size'])
        fanout[file] = np.array(sim[file]['fanout'])
        cache_hits[file] = None
        df_footprint[file] = None
        if sim[file]['sharding_file'] == '':
            name[file] = sim[file]['sharding_mode']
        elif 'sharding_name' in sim[file]:
            name[file] = sim[file]['sharding_name']
        else:
            name[file] = sim[file]['sharding_file']
            
        if 'cache_hits' in sim[file]:
            cache_hits[file] = np.array(sim[file]['cache_hits'])
            cache_refs[file] = np.array(sim[file]['cache_refs'])
            cache_sizes[file] = np.array(sim[file]['cache_sizes'])
            if 'cache_footprint' in sim[file]:
                cache_footprint[file] = sim[file]['cache_footprint']
                df_footprint[file] = pd.DataFrame(cache_footprint[file])
                df_footprint[file] = df_footprint[file].reindex(index=df_footprint[file].index[::-1])
            

    # calculating averages
    avg = lambda x: (x * np.arange(len(x))).sum() / x.sum()
    avg_fanout = dict()
    avg_out_packets = dict()
    avg_out_lookups = dict()
    avg_packet_size = dict()
    for file in input_files:
        avg_fanout[file] = avg(fanout[file])
        avg_out_packets[file] = avg(outgoing_packets[file])
        avg_out_lookups[file] = avg(outgoing_lookups[file])
        avg_packet_size[file] = avg(packet_size[file])

    # calculating load imbalance
    dispersion = lambda x: x.std() / x.mean()
    packets_imb = dict()
    lookups_imb = dict()
    for file in input_files:
        packets_imb[file] = dispersion(packets[file].sum(axis=0))
        lookups_imb[file] = dispersion(lookups[file].sum(axis=0))
    

    # plotting results
    
    if 'PM' in args.which_plots.split(','):
        fig, axs = plt.subplots(ncols=n_files+1,
            gridspec_kw={'width_ratios':[1]*n_files+[0.05]}, squeeze=True)
        fig.suptitle('Packets matrix')
        vmax = max([packets[file].max() for file in input_files])
        for i, file in enumerate(input_files):
            im = axs[i].imshow(packets[file], vmin=0, vmax=vmax)
            axs[i].set_xticks(range(P))
            axs[i].set_xticklabels(range(P), rotation=90)
            axs[i].set_yticks(range(P))
            axs[i].set_yticklabels(range(P))
            axs[i].set(title=name[file])
        fig.colorbar(im, cax=axs[-1])
        fig.tight_layout()
        if args.save:
            fig.savefig(name + '_PM.png')

    if 'LM' in args.which_plots.split(','):
        fig, axs = plt.subplots(ncols=n_files+1,
            gridspec_kw={'width_ratios':[1]*n_files+[0.05]}, squeeze=True)
        fig.suptitle('Lookups matrix')
        vmax = max([lookups[file].max() for file in input_files])
        for i, file in enumerate(input_files):
            im = axs[i].imshow(lookups[file], vmin=0, vmax=vmax)
            axs[i].set_xticks(range(P))
            axs[i].set_xticklabels(range(P))
            axs[i].set_yticks(range(P))
            axs[i].set_yticklabels(range(P))
            axs[i].set(title=name[file])
        fig.colorbar(im, cax=axs[-1])
        fig.tight_layout()
        if args.save:
            fig.savefig(name + '_LM.png')

    if 'RP' in args.which_plots.split(','):
        fig, axs = plt.subplots(ncols=n_files, squeeze=True)
        fig.suptitle(f'Received packets')
        top_lim = max([packets[file].sum(axis=0).max() for file in input_files])
        for i, file in enumerate(input_files):
            axs[i].bar(range(P), packets[file].sum(axis=0))
            axs[i].set_xticks(range(P))
            axs[i].set_xticklabels(range(P), rotation=90)
            axs[i].set_ylim(0, top_lim)
            axs[i].set(xlabel='Processor')
            axs[i].set(title=f'{name[file]}\nimbalance={packets_imb[file]:.3}')
        axs[0].set(ylabel='Number of received packets')
        for ax in axs[1:]:
            ax.set_yticks([])
            ax.set_yticklabels([])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        if args.save:
            fig.savefig(name + '_RP.png')

    if 'LR' in args.which_plots.split(','):
        fig, axs = plt.subplots(ncols=n_files, squeeze=True)
        fig.suptitle(f'Lookup requests')
        top_lim = max([lookups[file].sum(axis=0).max() for file in input_files])
        for i, file in enumerate(input_files):
            axs[i].bar(range(P), lookups[file].sum(axis=0))
            axs[i].set_xticks(range(P))
            axs[i].set_xticklabels(range(P), rotation=90)
            axs[i].set_ylim(0, top_lim)
            axs[i].set(xlabel='Processor')
            axs[i].set(title=f'{name[file]}\nimbalance={lookups_imb[file]:.3}')
        axs[0].set(ylabel='Number of lookup requests')
        for ax in axs[1:]:
            ax.set_yticks([])
            ax.set_yticklabels([])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        if args.save:
            fig.savefig(name + '_LR.png')


    if 'F' in args.which_plots.split(','):
        fig, axs = plt.subplots(ncols=n_files, squeeze=True)
        fig.suptitle(f'Fanout distribution')
        top_lim = max([fanout[file].max() for file in input_files])
        for i, file in enumerate(input_files):
            axs[i].bar(range(1,P+1), fanout[file][1:])
            axs[i].set_xticks(range(1,P+1))
            axs[i].set_xticklabels(range(1,P+1), rotation=90)
            axs[i].set_ylim(0, top_lim)
            axs[i].set(xlabel='Fanout')
            axs[i].set(title=f'{name[file]}\navg={avg_fanout[file]:.3}')
        axs[0].set(ylabel='Count')
        for ax in axs[1:]:
            ax.set_yticks([])
            ax.set_yticklabels([])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        if args.save:
            fig.savefig(name + '_F.png')


    if 'OP' in args.which_plots.split(','):
        fig, axs = plt.subplots(ncols=n_files, squeeze=True)
        fig.suptitle(f'Outgoing packets distribution')
        top_lim = max([outgoing_packets[file].max() for file in input_files])
        for i, file in enumerate(input_files):
            axs[i].bar(range(P), outgoing_packets[file])
            axs[i].set_xticks(range(P))
            axs[i].set_xticklabels(range(P), rotation=90)
            axs[i].set_ylim(0, top_lim)
            axs[i].set(xlabel='Packets')
            axs[i].set(title=f'{name[file]}\navg={avg_out_packets[file]:.3}')
        axs[0].set(ylabel='Count')
        for ax in axs[1:]:
            ax.set_yticks([])
            ax.set_yticklabels([])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        if args.save:
            fig.savefig(name + '_OP.png')

    if 'OL' in args.which_plots.split(','):
        fig, axs = plt.subplots(ncols=n_files, squeeze=True)
        fig.suptitle(f'Outgoing lookups distribution')
        top_lim = max([outgoing_lookups[file].max() for file in input_files])
        for i, file in enumerate(input_files):
            axs[i].bar(range(D+1), outgoing_lookups[file])
            axs[i].set_xticks(range(0,D+1,4))
            axs[i].set_xticklabels(range(0,D+1,4), rotation=90)
            axs[i].set_ylim(0, top_lim)
            axs[i].set(xlabel='Lookups')
            axs[i].set(title=f'{name[file]}\navg={avg_out_lookups[file]:.3}')
        axs[0].set(ylabel='Count')
        for ax in axs[1:]:
            ax.set_yticks([])
            ax.set_yticklabels([])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        if args.save:
            fig.savefig(name + '_OL.png')

    if 'OT' in args.which_plots.split(','):
        fig, axs = plt.subplots(ncols=n_files+1,
            gridspec_kw={'width_ratios':[1]*n_files+[0.05]}, squeeze=True)
        fig.suptitle('Outgoing tables matrix')
        vmax = max([outgoing_tables[file].max() for file in input_files])
        for i, file in enumerate(input_files):
            if args.reordering:
                im = axs[i].imshow(outgoing_tables[file][:, reordering], vmin=0, vmax=vmax)
                axs[i].set_xticks(range(D))
                axs[i].set_xticklabels(reordering, rotation=90)
            else:
                im = axs[i].imshow(outgoing_tables[file], vmin=0, vmax=vmax)
                axs[i].set_xticks(range(D))
                axs[i].set_xticklabels(range(D), rotation=90)
            axs[i].set_yticks(range(P))
            axs[i].set_yticklabels(range(P))
            axs[i].set(xlabel='Table index')
            axs[i].set(title=name[file])
        axs[0].set(ylabel='Processor')
        fig.colorbar(im, cax=axs[-1])
        fig.tight_layout()
        if args.save:
            fig.savefig(name + '_OT.png')

    if 'PS' in args.which_plots.split(','):
        fig, axs = plt.subplots(ncols=n_files, squeeze=True)
        fig.suptitle(f'Packet size distribution')
        top_lim = max([packet_size[file].max() for file in input_files])
        for i, file in enumerate(input_files):
            axs[i].bar(range(1,D+1), packet_size[file][1:])
            axs[i].set_xticks(range(1,D+1))
            axs[i].set_xticklabels(range(1,D+1), rotation=90)
            axs[i].set_ylim(0, top_lim)
            axs[i].set(xlabel='Packet size')
            axs[i].set(title=f'{name[file]}\n' +
                f'avg={avg_packet_size[file]:.3}')
            #   f'total={packets[file].sum()}')
        axs[0].set(ylabel='Count')
        for ax in axs[1:]:
            ax.set_yticks([])
            ax.set_yticklabels([])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        if args.save:
            fig.savefig(name + '_PS.png')


    if cache_hits is not None:
        if 'C' in args.which_plots.split(','):
            fig, axs = plt.subplots(ncols=n_files+1,
            gridspec_kw={'width_ratios':[1]*n_files+[0.05]}, squeeze=True)
            fig.suptitle('Cache hit-rates')
            vmax = max([(cache_hits[file] / cache_refs[file]).max() for file in input_files])
            for i, file in enumerate(input_files):
                if args.reordering:
                    im = axs[i].imshow((cache_hits[file] / cache_refs[file])[:, reordering], vmin=0, vmax=vmax)
                    axs[i].set_xticks(range(D))
                    axs[i].set_xticklabels(reordering, rotation=90)
                else:
                    im = axs[i].imshow((cache_hits[file] / cache_refs[file]), vmin=0, vmax=vmax)
                    axs[i].set_xticks(range(D))
                    axs[i].set_xticklabels(range(D), rotation=90)
                axs[i].set_yticks(range(P))
                axs[i].set_yticklabels(range(P))
                axs[i].set(xlabel='Table index')
                axs[i].set(title=name[file])
            axs[0].set(ylabel='Processor')
            fig.colorbar(im, cax=axs[-1])
            fig.tight_layout()
            if args.save:
                fig.savefig(name + '_C.png')

        if 'CT' in args.which_plots.split(','):
            fig, axs = plt.subplots(ncols=n_files, squeeze=True)
            fig.suptitle('Cache hit-rates by table')
            vmax = max([(cache_hits[file].sum(axis=0) / cache_refs[file].sum(axis=0)).max() for file in input_files])
            for i, file in enumerate(input_files):
                axs[i].bar(range(D), (cache_hits[file].sum(axis=0) / cache_refs[file].sum(axis=0))[reordering])
                top_lim = max(top_lim, axs[i].get_ylim()[1])
                axs[i].set_xticks(range(D+1))
                axs[i].set_xticklabels(range(D+1), rotation=90)
                axs[i].set_ylim(0, top_lim)
                axs[i].set(xlabel='Table index')
                axs[i].set(title=name[file])
            axs[0].set(ylabel='Hit-rates')
            for ax in axs[1:]:
                ax.set_yticks([])
                ax.set_yticklabels([])
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.tight_layout()
            if args.save:
                fig.savefig(name + '_CT.png')

    if not args.save:
        plt.show()

# example
# python comparer.py -r -p OP,OL,RP,LM -i results\sim_20210422-093501.json,results\sim_20210422-093926.json,results\sim_20210424-114829.json,results\sim_20210425-104550.json
