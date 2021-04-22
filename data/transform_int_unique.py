import pandas
import numpy
from time import time
import sys

input=sys.argv[1]
output=sys.argv[2]

nrows = int(1e6)
# columns = range(14, 40)
columns = [33, 14, 35, 23, 34]

print('Reading ... ', end='', flush=True)
t = time()
data = (
pandas
.read_csv(input, header=None, sep='\t', usecols=sorted(columns), compression='gzip', nrows=nrows)
.replace(numpy.nan, '0')
)
t = time() - t
print(t)

print('Transforming ... ', end='', flush=True)
t = time()
for col in data.columns:
    data[col] = data[col].apply(lambda x: int(hex(col)+x, 16))
t = time() - t
print(t)


print('Exporting ... ', end='', flush=True)
t = time()
data[columns].to_csv(output, header=False, index=False, sep=' ')
t = time() - t
print(t)
