import numpy as np
from glob import glob
from time import time
import sys
import re

n_tables = 26

def empty_list_init(d, r):
    for i in r:
        d[i] = []

# returns a dictionary where keys are table ids
def get_content(file, p):
    res = dict()
    # initialization
    empty_list_init(res, range(0,n_tables))
    
    with open(file, 'r') as f:
        f.readline() # skip
        f.readline() # skip
        line = f.readline()
        while line:
            i = int(line)
            if i <= 0xff0:
                table_id = i // 16 -14
                emb_id = 0
            else:
                n = hex(i)[2:]
                n_head, n_body = n[:-8], n[-8:]
                if len(n_head) == 0:
                    n_head = 'e'
                table_id = int(n_head, 16) -14
                emb_id = int(n_body, 16)
            try:
                res[table_id].append(f'{emb_id}:{p}')
            except:
                print('dio porco')
                print(table_id, emb_id, n, line)
            line = f.readline()
    return res

input = sys.argv[1]
output = sys.argv[2]

all = glob(input + '_partition_*')

print('Found:')
print(*all, sep='\n')
print()

S = dict()
empty_list_init(S, range(0,n_tables))

t = time()
for file in all:
    p = re.findall(f'\d+', file)[-1]
    print('Processing partition ', p)
    res = get_content(file, p)

    # merging
    for i in range(0,n_tables):
        S[i] += res[i]

t = time() - t
print('Processing time: ', t)

# exporting
print('Exporting ... ', end='', flush=True)
t = time()
with open(output, 'w') as f:
    for i in range(0,n_tables):
        f.write(','.join(S[i]) + '\n')
t = time() - t
print(t)
    
