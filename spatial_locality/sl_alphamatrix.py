# Calulates, plots and saves the alpha correlation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product, combinations
from time import time

input = '..\data\day_100k.csv'
print(f'Reading \'{input}\' ... ', end='', flush=True)
selected_columns = range(14, 14+26)
t = -time()
data = pd.read_csv(input, sep='\t', header=None, usecols=selected_columns)
t += time()
print(f'{t:.5} sec')

data = data.replace(np.nan, '0')
N = len(data)

# for each pair of sparse feature indexes
hcs = []
print('Counting unique values ... ', end='', flush=True)
t = -time()
counts = [data[i].value_counts().size for i in selected_columns]
t += time()
print(f'{t:.5} sec')

# processing features 
A = np.empty( (len(selected_columns), len(selected_columns)) )
A[:] = np.nan
offset = selected_columns[0]
t = -time()
for i,j in combinations(selected_columns, r=2):
    if i == j:
        continue
    print(f'\rProcessing alpha({i},{j}) ... ', end='', flush=True)
    L = data[[i,j]].value_counts().size
    m = min(counts[i-offset] * counts[j-offset], N)
    M = max(counts[i-offset], counts[j-offset])
    #M = counts[j-offset]
    corr = (m-L) / (m-M)
    A[i-offset,j-offset] = corr
t += time()
print('\r',*([' ']*80), sep='', end='') # clearing 80 chars of the last line
print(f'\rDone in {t:.8} sec')

# saving to file
outname = 'alpha_' + str(int(time()))
np.save(outname, A)

# plotting image
plt.imshow(A)
plt.xticks(range(len(selected_columns)), selected_columns)
plt.yticks(range(len(selected_columns)), selected_columns)
plt.colorbar()
plt.grid(True)
plt.tight_layout()
plt.savefig(outname + '.png')
plt.show()



