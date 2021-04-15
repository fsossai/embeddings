# Calculates, plots and saves the alpha correlation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from time import time
import sys

# return a dictionary indexed by pair of columns
def alpha_correlation(data):
    N = len(data)
    cols = data.columns

    # getting the cardinality of each column
    nunique = dict()
    for col in cols:
        nunique[col] = data[col].value_counts().size

    # calculating alpha correlation for each pair
    A = dict()
    for i,j in combinations(cols, r=2):
        t = data[[i,j]].value_counts().size
        M = min(nunique[i] * nunique[j], N)
        m = max(nunique[i], nunique[j])
        A[i,j] = (M - t) / (M - m)
    
    return A


if __name__ == '__main__':
    t = time()

    # reading input
    data = pd.read_csv(sys.argv[1], header=None)
    A = alpha_correlation(data)

    # creating a matrix suitable for a DataFrame
    matrix = []
    for i,j in A:
        matrix.append([i, j, A[i,j]])
    df = pd.DataFrame(matrix, columns=['i','j','alpha'])

    # sorting and saving
    df.sort_values(by='alpha', ascending=False, inplace=True)
    df.to_csv('alpha.csv', index=False)
    
    t = time() - t
    print(f'Elapsed time: {t} sec')
