import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import argparse
from itertools import groupby, zip_longest

def zip_discard(*iterables):
    return map(
            lambda x: filter(lambda y: y is not None, x),
            zip_longest(*iterables)
    )

class MemoryDevices:
    def __init__(self, ndevices, aggregate_size):
        self.ndevices = ndevices
        self.aggregate_size = aggregate_size
        self.local_capacity = int(self.aggregate_size / self.ndevices)
        self.reset()

    def reset(self):
        self.counter = 0
        self.requests = []
        self.memories = [set() for i in range(self.ndevices)]
        self.ids = 0

    def load_random(self, dataset):
        for i in dataset.columns:
            for id,_ in dataset[i].value_counts().iteritems():
                self.memories[np.random.randint(0,self.ndevices)].add( (i,id) )
        self.ids += len(dataset)

    def load_greedy(self, dataset):
        local_free_space = self.local_capacity
        current_device = 0
        for col in dataset.columns:
            counts = data[col].value_counts()
            remaining = counts.size
            added = 0
            while remaining > 0:
                if local_free_space == 0:
                    if current_device == self.ndevices-1:
                        raise Exception('out of space!')
                    else:
                        current_device += 1
                        local_free_space = self.local_capacity
                selected = min(local_free_space, remaining)
                
                # loading the selected embeddings
                for id,_ in counts[added:added+selected].iteritems():
                    self.memories[current_device].add( (col,id) )

                added += selected
                local_free_space -= selected
                remaining -= selected
    
    def load_roundrobin(self, dataset):
        def emb_tuples(col, items):
            for x,_ in items:
                yield (col,x)
        def vcounts():
            for col in dataset.columns:
                yield emb_tuples(col, dataset[col].value_counts().iteritems())
        
        # adding 
        local_free_space = self.local_capacity
        current_device = 0
        for x in zip_discard(*vcounts()):
            vals = list(x)
            remaining = len(vals)
            added = 0
            while remaining > 0:
                if local_free_space == 0:
                    if current_device == self.ndevices-1:
                        raise Exception('out of space!')
                    else:
                        current_device += 1
                        local_free_space = self.local_capacity
                selected = min(local_free_space, remaining)
                
                # loading the selected embeddings
                for val in vals:
                    self.memories[current_device].add(val)
                #self.memories[current_device] += vals[added:added+selected] #for lists

                added += selected
                local_free_space -= selected
                remaining -= selected
        

    def load_alpha(self, dataset, corr_file):
        raise Exception('ERROR: Alpha strategy not implemented yet')
        ## sharding embeddings according to A
        #self.A = np.load(corr_file)        
        #A = self.A
        #at = lambda i,j: (min(i,j), max(i,j))

    def random_strategy(devices, features):
        return [
            n * (1 - np.power(1 - 1/n, features))
            for n in devices
        ]

    def print_status(self):
        for i,mem in enumerate(self.memories):
            print(f'MD{i}\t: {len(mem)}\t{len(mem)/self.local_capacity*100:.4}%')

    def lookup(self, query):
        self.requests.append(
            len(set([
                next(
                    j for j,s in enumerate(self.memories)
                    if (i,q) in s
                )
                for i,q in query.iteritems()
            ]))
        )
    
    def avg_fanout(self):
        return np.mean(self.requests)    

def simulation(devices, queries, strategy, aggregate_size):
    print_step = 500
    mem = MemoryDevices(devices, aggregate_size)

    # loading query according to a given strategy
    print('Loading to distributed memory ... ', end='', flush=True)
    t = time()
    if strategy == 'greedy':
        mem.load_greedy(queries)
    elif strategy == 'alpha':
        mem.load_alpha(queries, 'corr_matrix_symm.npy')
    elif strategy == 'roundrobin':
        mem.load_roundrobin(queries)
    t = time() - t
    print(f'{t:.5} sec')

    #print(*list(mem.memories[0])[0:20],sep='\n')
    #raise Exception('!!!')

    #mem.print_status()
    # simulating all lookups
    n = len(queries)
    print('Processing ... ', end='', flush=True)
    t = time()
    for i,q in queries.iterrows():
        mem.lookup(q)
        if (i % print_step == 0):
            print(f'\rProcessing\t: {i/n * 100:.4}%', end='')
    t = time() - t
    fanout = mem.avg_fanout()

    print(f'\rProcessing\t: 100% ... {t:.5} sec')
    print(f'Average fanout\t: {fanout:.3}')
    return fanout


# Main ------------------------------------------------------------------------

total_time = time()
parser = argparse.ArgumentParser()
parser.add_argument('--random', action='store_true', default=False)
parser.add_argument('--greedy', action='store_true', default=False)
parser.add_argument('--roundrobin', action='store_true', default=False)
parser.add_argument('--alpha', action='store_true', default=False)
args = parser.parse_args()

dataset = pd.read_csv('..\data\day_23.gz',
    compression=None,
    header=None, sep='\t',
    chunksize=int(1e4))
data = next(dataset)
data = data.replace(np.nan, '0')

queries = data[range(14,14+26)]
print(f'Number of queries\t: {len(queries)}')

devices_per_sim = np.power(2, range(1,11))
results = {}

# parsing arguments
strategies = []
if args.greedy:
    strategies.append('greedy')
    results['greedy'] = []
if args.alpha:
    strategies.append('alpha')
    results['alpha'] = []
if args.roundrobin:
    strategies.append('roundrobin')
    results['roundrobin'] = []

# running all simulations
aggregate_size = int(2*1e5) # 2*1e5 for 1e5 queries
M = len(devices_per_sim)
for i,devices in enumerate(devices_per_sim):
    print(f'\n[*] Simulation {i+1}/{M},',
        f'{devices} devices,')
    for strategy in strategies:
        fanout = simulation(devices, queries, strategy, aggregate_size)
        results[strategy].append( fanout )

# saving simulation results to a numpy file
labels = {
    strategy : results[strategy]
    for strategy in strategies
}
index = str(int(time()))
np.savez('fanout_' + index, **labels)

for strategy in strategies:
    plt.plot(devices_per_sim, results[strategy], '.-', label=strategy)

# plotting theoretical fanout for a random strategy
plt.plot(devices_per_sim,
    MemoryDevices.random_strategy(devices_per_sim, features=26),
    '.--', label='random'
)

total_time = time() - total_time
print(f'\nTotal elapsed time\t: {total_time:.5} sec')

plt.grid(True)
plt.title(f'Average Fanout vs Memory Devices, {len(queries)} queries')
plt.xlabel('Number of memory devices')
plt.ylabel('Average fanout')
plt.legend()
plt.savefig('fanout_' + index + '.png')
plt.show()



