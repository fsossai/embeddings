import argparse
from time import time
import simtools
import sys
sys.path.append('..')
import bigdatatools

parser = bigdatatools.default_parser()
parser.description = 'Embedding Tables creator. Creates .bin files ' \
                     'to be processed by a simulator'
parser.add_argument('--files', '-f', type=str, required=True)
parser.add_argument('--output', '-o', type=str, default=None)
args = parser.parse_args()

if args.selected_columns is not None:
    selected_columns = bigdatatools.get_range_list(args.selected_columns)
else:
    selected_columns = None

pandas_kwargs = {
    'sep': '\t',
    'header': None,
    'usecols': selected_columns,
    'chunksize': args.chunk_size,
    'compression': 'gzip' if args.gzip else None
}

outname = args.output or 'embtables' + str(int(time())) + '.bin'

simtools.create_tables_from_dataframe(
    inputfiles=args.files,
    outputfile=outname,
    nchunks=args.n_chunks,
    selected_columns=selected_columns,
    **pandas_kwargs
)
