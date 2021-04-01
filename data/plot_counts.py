import pandas as pd
import matplotlib.pyplot as plt
from time import time
import argparse


def setup_histogram(nuniques, highlighted, highlight, ticks, linear_plot, top):
    fig, ax = plt.subplots()
    if highlight == 1:
        h_label = f'Fraction of IDs appearing only once'
    elif highlight > 1:
        h_label = f'Fraction of IDs appearing no more than {highlight} times'
    else:
        h_label = None

    nuniques = nuniques[:top]
    highlighted = highlighted[:top]
    ticks = ticks[:top]
    base = [n - h for n, h in zip(nuniques, highlighted)]
    nsel = len(base)
    ax.bar(range(nsel), base,
        log=not linear_plot)
    ax.bar(range(nsel), highlighted,
        log=not linear_plot, bottom=base, label=h_label)
    
    plt.xticks(range(nsel), ticks, rotation=70)
    plt.xlabel('Feature index')
    plt.ylabel('Number of unique IDs')
    plt.title("Cardinality of the features' domain" +
        f" (Top {top})" if top is not None else "")

    plt.tight_layout()
    if highlight > 0:
        plt.legend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'Cardinality of categorical features\' domain'
    parser.add_argument('input', type=str)
    parser.add_argument('--top', '-t', type=int, default=None)
    parser.add_argument('--highlight', '-H', type=int, default=0)
    parser.add_argument('--linear-plot', '-l', action='store_true', default=False)
    parser.add_argument('--reverse-order', '-r', action='store_true', default=False)
    parser.add_argument('--out-name', '-o', type=str, default=None, required=False)
    
    args = parser.parse_args()

    timestamp = str(time())
    out_name = args.out_name or 'counts_{timestamp}'

    # reading CSV file
    df = pd.read_csv(args.input)

    nuniques = df['nunique'].to_list()
    highlighted = df['highlighted'].to_list()

    if args.reverse_order:
        setup_histogram(
            nuniques=list(reversed(df['nunique'].to_list())),
            highlighted=list(reversed(df['highlighted'].to_list())),
            highlight=1,
            ticks=list(reversed(df['table'])),
            linear_plot=args.linear_plot,
            top=args.top
        )
    else:
        setup_histogram(
            nuniques=df['nunique'].to_list(),
            highlighted=df['highlighted'].to_list(),
            highlight=1,
            ticks=df['table'],
            linear_plot=args.linear_plot,
            top=args.top
        )
    
    if args.out_name is not None:
        plt.savefig(out_name + '.png')
    plt.show()

# Command line example:
# python plot_counts.py -l -t 10 -H 1 -r counts.csv
