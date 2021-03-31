# Performance profiler, August 2020
# Author: Federico Sossai
# This software is freely redistributable
#
# Scientific reference:
#   Dolan E.D., Moŕe J.J.
#   Benchmarking Optimization Software with Performance Profiles.
#   Mathematical Programming 91(2):201–213
#   2002
#
# Usage: perfprofiler.py [options] <input>

import csv
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import optparse 


# Adding avaible options to the parser

parser = optparse.OptionParser("perfprofiler.py <input> [options]",
    description="<input> is a file in CSV format. The first row must contain the number of \
actual data columns to be considered. Example: '2, column1, column2'. The following lines \
will contain the name of the run followed by two comma separated values, like 'inst1, 123.4, 567.8'")

parser.add_option('-x','--xlabel', action="store", dest="xlabel", default="Ratio",
                    help="Set a label for the X axis.")
parser.add_option('-y','--ylabel', action="store", dest="ylabel", default="Percentage",
                    help="Set a label for the Y axis.")
parser.add_option('-t','--title', action="store", dest="title", default="Performance profile",
                    help="Set the plot title.")
parser.add_option('-m','--marker-type', action="store", dest="markertype", default="points",
                    help="[letters|symbols|points|none]. Set a marker type to easily distinguish lines. "
                    "Default: points.")
parser.add_option('-l','--limit', action="store", dest="limit", default="",
                    help="Set a x-axis' right-most (resp. left-most) limit of a minimization \
                    (resp. maximization) problem.")
parser.add_option('-o','--output', action="store", dest="output", default="",
                    help="Output file name. The plot can be saved in different formats.")
parser.add_option('-p','--problem-type', action="store", dest="ptype", default="min",
                    help="[max|min]. Maximization or Minimization type of problem. "
                    "Default: min.")
options, args = parser.parse_args()


# Checking arguments provided in the command line

if len(args) == 0:
    sys.exit("ERROR: An input file must be provided. Type '--help' for more info.")
elif len(args) > 1:
    print("WARNING: " + str(len(args) - 1) + " unrecognized argument/s: '" + "', '".join(args[1:]) + "'")
if options.markertype == 'letters':
    markers = ['$' + l + '$' for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
elif options.markertype == 'symbols':
    markers = [m for m in matplotlib.markers.MarkerStyle.markers if m != "None"]
elif options.markertype == 'points':
    markers = ['.']
elif options.markertype == 'none':
    markers = ['None']
else:
    sys.exit("ERROR: invalid marker type. Available: [letters|symbols|points|none].")

if options.ptype == 'max':
    get_best = np.max
elif options.ptype == 'min':
    get_best = np.min
else:
    sys.exit("ERROR: invalid problem type: Available [max|min].")

# Reading CSV input 

with open(args[0], newline='') as f:
    table = list(csv.reader(f))
    methods = int(table[0][0])


# Computing the so called 'performance ratios'

runs = len(table) - 1
all = np.zeros((runs,methods))
for index,row in enumerate(table[1:]):
    vals = np.array([float(i) for i in row[1:methods+1] if (len(i) > 0)])
    vals = vals / get_best(vals)
    all[index, :] = vals


# Creating every line in the plot iteratively

reverse=True if options.ptype == 'max' else False
for i in range(0, methods):
    selected = np.random.randint(0, len(markers))
    plt.step(sorted(all[:,i], reverse=reverse), np.arange(1, runs + 1) / runs,
            where='post',
            marker=markers[selected],
            label=table[0][i+1])
    if len(markers) > 1:
        del markers[selected]


# Plotting of saving output to file

plt.grid(True)
plt.xlabel(options.xlabel)
plt.ylabel(options.ylabel)
plt.yticks(
    [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
)
plt.title(options.title)
plt.legend()
if len(options.limit) > 0:
    if options.ptype == 'max':
        plt.xlim(left=float(options.limit))
    elif options.ptype == 'min':
        plt.xlim(right=float(options.limit))
if len(options.output) == 0:
    plt.show()
else:
    plt.savefig(options.output)
