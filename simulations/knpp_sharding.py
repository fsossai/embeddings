import pandas as pd
import numpy as np
from time import time
import heapq
import sys

def greedy(number_list, k=2):
    heap = [(0, []) for i in range(k)]
    for n in sorted(number_list, reverse=True, key=lambda x: x[1]):
        cost, elements = heap[0]
        elements.append(n)
        new_group = (cost + n[1], elements)
        heapq.heapreplace(heap, new_group)
    heap.sort(key=lambda x: x[0])
    return [y for x, y in heap]

def get_index_and_id(number):
    h = hex(number)[2:]
    index = h[:-8]
    emb_id = h[-8:]
    return int(index, 16) - 1, int(emb_id, 16)

input_file = '..//data//day_23_1M_offset1.hypergraph'
output_file = f's_knnp{int(time())}.txt'
P = 16

print('Importing and processing dataset ... ', end='', flush=True)
t = time()
data = pd.read_csv(input_file, header=None, sep=' ')
vc = [data[col].value_counts() for col in data.columns]
t = time() - t
print(t)

print('Sharding ... ', end='', flush=True)
t = time()
T = [[] for _ in range(P)]
for col in data.columns:
    for p, content in enumerate(greedy(vc[col].iteritems(), P)):
        T[p] += content
t = time() - t
print(t)

# exporting
print('Exporting ... ', end='', flush=True)
t = time()
tables = [[] for _ in data.columns]
for p, table in enumerate(T):
    for emb in table:
        index, emb_id = get_index_and_id(emb[0])
        tables[index].append(f'{emb_id}:{p}')

with open(output_file, 'w') as f:
    for table in tables:
        f.write(','.join(table) + '\n')
t = time() - t
print(t)