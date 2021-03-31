@echo off
echo Generating the performance profile plot

python perfprofiler.py                                          ^
 stack_simulator\comparison.csv                                 ^
 --xlabel           "Ratio"                                     ^
 --ylabel           "Percentage"                                ^
 --title            "Policies comparison - Performance profile" ^
 --output           perfprof.png                                ^
 --problem-type     max                                         ^
 --marker-type      points

echo Done