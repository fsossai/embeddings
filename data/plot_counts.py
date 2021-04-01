import pandas as pd
import matplotlib.pyplot as plt
from time import time
import argparse


def setup_histogram(nuniques, highlighted, highlight, ticks, linear_plot,
    top, use_percentage=False):
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
    if use_percentage:
        base = [x / y for x, y in zip(base, nuniques)]
        highlighted = [x / y for x, y in zip(highlighted, nuniques)]
    nsel = len(base)

    ax.bar(range(nsel), base,
        log=not linear_plot)
    ax.bar(range(nsel), highlighted,
        log=not linear_plot, bottom=base, label=h_label)
    
    plt.xticks(range(nsel), ticks, rotation=70)
    plt.xlabel('Feature index')
    if use_percentage:
        plt.ylabel('Percentage of unique IDs')
        plt.yticks(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ) # np.arange no thanks
    else:
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
    parser.add_argument('--out-name', '-o', type=str, default=None)
    parser.add_argument('--use-percentage', '-p', action='store_true', default=False)
    
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
            top=args.top,
            use_percentage=args.use_percentage
        )
    else:
        setup_histogram(
            nuniques=df['nunique'].to_list(),
            highlighted=df['highlighted'].to_list(),
            highlight=1,
            ticks=df['table'],
            linear_plot=args.linear_plot,
            top=args.top,
            use_percentage=args.use_percentage
        )
    
    if args.out_name is not None:
        plt.savefig(out_name + '.png')
    plt.show()

# Command line example:
# python plot_counts.py -l -t 10 -H 1 -r -p counts.csv
