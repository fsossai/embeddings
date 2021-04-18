import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

nu = [188070,13734,12134,6089,16175,3,6239,1170,37,142314,38457,36324,10,1889,5862,68,4,851,14,197990,86250,171445,30741,8138,54,33]
filtering_level = 0
alpha_filtering_level = 0.0

def find_corresponding_right_index(s, key):
    i = s.index.names[0]
    return list(
        y
        for x, y in 
        s.iloc[s.index.get_level_values(level=i) == key]
        .sort_values(ascending=False)
        .index
    )

def build_snake(path, vcx, starter, assigned):
    it = iter(path)
    next(it)
    snake = []
    last = starter
    snake.append(last)
    assigned.add((path[0], last))

    for i, j in zip(path, it):
        connected = find_corresponding_right_index(vcx[i,j], last)
        for r in connected:
            if (j, r) not in assigned:
                assigned.add((j, r))
                snake.append(r)
                last = r
                break
    return snake

def find_suitable(i, items, assigned):
    return next((
            item for item in items
            if (i, item) not in assigned
        ), None)

def send_to_p(path, snake, p, lookup_tables):
    for table, id in zip(path, snake):
        lookup_tables[table].append((id,p))

A = pd.read_csv('alpha.csv')
mat = dict()
for _,t in A.iterrows():
    i,j,a = t.to_list()
    mat[i,j] = a

# filtering
A = A[A['alpha'] >= alpha_filtering_level]
#print(A)
# identifying the paths
path_code = dict()
n_path_codes = 0
for _, t in A.iterrows():
    i, j, _ = t.to_numpy(dtype=np.int)
    if (i not in path_code) and (j not in path_code):
        path_code[i] = n_path_codes
        path_code[j] = n_path_codes
        n_path_codes += 1
    elif (i in path_code) and (j not in path_code):
        path_code[j] = path_code[i]
    elif (i not in path_code) and (j in path_code):
        path_code[i] = path_code[j]

# putting paths in a more readable format
paths = {i:[] for i in range(n_path_codes)}
for i in path_code:
    paths[path_code[i]].append(i)

# sorting by table size and trasforming dict into list
paths = [
    sorted([(x, nu[x]) for x in path], reverse=True, key=lambda x: x[1])
    for path in paths.values()
]

# sorting by size of head table
paths = sorted(paths, reverse=True, key=lambda x: x[0][1])

# printing all paths
for i, path in enumerate(paths):
    print(f'===> Path {i}, len={len(path)}:')
    print(*path, sep=' ', end='\n\n')

# printing all columns involved
all_cols = sorted(list(path_code))
print(f'===> All {len(path_code)} columns involved:')
print(*all_cols, sep=' ', end='\n\n')

# converting paths into a dict of column indexes
paths = [
    list(map(lambda x: int(x[0]), path))
    for path in paths
]

'''
===> Path 0, len=6:
(19, 197990) (0, 188070) (21, 171445) (9, 142314) (20, 86250) (10, 38457)

===> Path 1, len=6:
(22, 30741) (1, 13734) (14, 5862) (7, 1170) (15, 68) (25, 33)

===> Path 2, len=4:
(2, 12134) (23, 8138) (6, 6239) (5, 3)
'''
 

# importing dataset
print('Reading dataset ... ', end='', flush=True)
data = pd.read_csv('cached\\p.csv', header=None)
print('OK')

# cross value_counts for every snake pair
print('Cross value_counts ... ', end='', flush=True)
vcx = dict()
for path in paths:
    it = iter(path)
    next(it) # consuming one item
    for i, j in zip(path, it):
        vcx[i,j] = data[[i,j]].value_counts()
# considering also all pairs (master,j)
heads = iter(next(zip(*paths)))
master_head = next(heads)
for h in heads:
    vcx[master_head, h] = data[[master_head, h]].value_counts()
print('OK')

# extracting value_counts of each column from vcx
print('Single value_counts ... ', end='', flush=True)
vc = dict()
for i, j in vcx:
    if i not in vc:
        vc[i] = vcx[i,j].sort_index(level=i).groupby(level=i).sum()
    if j not in vc:
        vc[j] = vcx[i,j].sort_index(level=j).groupby(level=j).sum()
print('OK')

# filtering out singlets from vc
print('Filtering level =', filtering_level)
print('Filter out singles ... ', end='', flush=True)
for i in vc:
    vc[i] = vc[i][vc[i] > filtering_level]
print('OK')

# random index reordering
print('Index reordering ... ', end='', flush=True)
master_head = paths[0][0]
master_ids = np.random.permutation(vc[master_head].index)
starters_gen = dict()
starters_gen[master_head] = iter(master_ids)
for path in paths:
    h = path[0]
    starters_gen[h] = iter(np.random.permutation(vc[h].index))
print('OK')

# Snake sharding
t = time()
P = 16 # fictious number of processors
K = vc[master_head].size # number of snakes to be selected for each path
assigned = set()

# initializing lookup tables
lookup_tables = dict()
for c in all_cols:
    lookup_tables[c] = []

selected = 0
iterations = 0
for k in range(K):
    p = np.random.randint(P)

    # building master path
    h = paths[0][0]
    starter = next(starters_gen[h], None)
    if starter is None:
        print('Reached end of master path')
        break
    snake = build_snake(paths[0], vcx, starter, assigned)
    send_to_p(paths[0], snake, p, lookup_tables)
    selected += len(snake)

    # building other paths
    for path in paths[1:]:
        h = path[0]
        starters = find_corresponding_right_index(
            vcx[master_head, h], starter)
        starter = (
            find_suitable(h, starters, assigned) or
            find_suitable(h, starters_gen[h], assigned)
        )

        if starter is not None:
            snake = build_snake(path, vcx, starter, assigned)
            selected += len(snake)
            send_to_p(path, snake, p, lookup_tables)
        
    if (k+1) % 1000 == 0:
        perc = (k+1)/K * 100
        print(f'===> Iteration {k+1}/{K}, {perc:.3} %', end='\t')
        print(f'selected: {selected}')
    iterations += 1

t = time() - t
perc = iterations/K * 100
print(f'===> Iteration {iterations}/{K}, {perc:.4} %', end='\t')
print(f'selected: {selected}')
print('Elapsed time:', t)
print()

# exporting to sharding table
print('Exporting to file ... ', end='', flush=True)
with open('alpha_s.txt', 'w') as f:
    for col in data.columns:
        line = []
        if col in lookup_tables:
            for id, p in lookup_tables[col]:
                line.append(f'{id}:{p}')
        if len(line) == 0:
            line = ['0:-1']
        f.write(','.join(line) + '\n')
print('OK')