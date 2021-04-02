import pandas
import numpy
import sys

input=sys.argv[1]
output=sys.argv[2]

(
pandas
.read_csv(input, header=None, sep='\t', usecols=range(14,40))
.replace(numpy.nan, '0')
.apply(lambda x: x.apply(lambda y: int(y, 16)))
.to_csv(output, header=False, index=False)
)