import sys
import numpy as np
from time import time
import matplotlib.pyplot as plt
import pandas as pd

input_file = sys.argv[1]
# P = int(sys.argv[2])
P = 16


print('Importing and processing dataset ... ', end='', flush=True)
t = time()
data = pd.read_csv('cached//p.csv', header=None)
vc = [data[col].value_counts() for col in data.columns]
t = time() - t
print(t)

counts = np.zeros((P,), dtype=np.int32)
weighted = np.zeros((P,), dtype=np.int32)

# reading file
print('Reading file ... ', end='', flush=True)
t = time()
with open(input_file, 'r') as f:
    for i, line in enumerate(f):
        table = line.split(',')
        for emb in table:
            s = emb.split(':')
            ID, p = int(s[0]), int(s[1])
            if p >= 0:
                counts[p] += 1
                weighted[p] += vc[i][ID]
            
t = time() - t
print(t)

print('Total size:', counts.sum())
print(*counts)

plt.figure(0)
plt.bar(range(P), counts)
plt.xticks(range(P), range(P), rotation=0)
plt.xlabel('Processor index')
plt.ylabel('Embeddings counts')
plt.tight_layout()

plt.figure(1)
plt.bar(range(P), weighted)
plt.xlabel('Processor index')
plt.ylabel('Number of expected requests')
plt.xticks(range(P), range(P))
plt.tight_layout()

plt.show()

