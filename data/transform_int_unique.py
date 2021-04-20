import pandas
import numpy
from time import time
import sys

input=sys.argv[1]
output=sys.argv[2]

print('Reading ... ', end='', flush=True)
t = time()
data = (
pandas
.read_csv(input, header=None, sep='\t', usecols=range(14,40), compression='gzip', nrows=int(1e6))
.replace(numpy.nan, '0')
)
t = time() - t
print(t)

print('Transforming ... ', end='', flush=True)
t = time()
for i, col in enumerate(data.columns):
    data[col] = data[col].apply(lambda x: int(hex(i)+x, 16))
t = time() - t
print(t)


print('Exporting ... ', end='', flush=True)
t = time()
data.to_csv(output, header=False, index=False, sep=' ')
t = time() - t
print(t)
