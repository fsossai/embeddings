@echo off

if not exist results (
    mkdir results
)

python strategies.py                                            ^
 --chunk-size               500                                 ^
 --n-chunks                 2                                   ^
 --strategies               greedy,roundrobin1,roundrobin2      ^
 --n-devices                2,4,8,16                            ^
 --queries-file             "..\data\day_*.gz"                  ^
 --gzip                                                         ^
 --column-selection         14-39                               ^
 --embedding-tables         "tables\day_23_1M.bin"              ^
 --output-dir               "results"
