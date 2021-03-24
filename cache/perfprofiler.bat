@echo off
echo Generating the performance profile plot

python perfprofiler.py                                          ^
 stack_simulator\comparison.csv                                 ^
 --xlabel           "Ratio"                                     ^
 --ylabel           "Percentage"                                ^
 --title            "LRU/LFU comparison - Performance profile"  ^
 --output           perfprof.pdf                                ^
 --problem-type     max                                         ^
 --marker-type      points                                      ^
 --limit            0.7

echo Done