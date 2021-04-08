@echo off
echo Generating the performance profile plot

<NUL set /p=max...
python perfprofiler.py                                          ^
 results\comparison.csv                                         ^
 --xlabel           "Ratio"                                     ^
 --ylabel           "Percentage"                                ^
 --title            "Policies comparison - Performance profile" ^
 --output           perfprof_max.png                            ^
 --problem-type     max                                         ^
 --marker-type      points

echo OK
<NUL set /p=maxr...
python perfprofiler.py                                          ^
 results\comparison.csv                                         ^
 --xlabel           "Ratio"                                     ^
 --ylabel           "Percentage"                                ^
 --title            "Policies comparison - Performance profile" ^
 --output           perfprof_maxr.png                           ^
 --problem-type     maxr                                        ^
 --marker-type      points

echo OK

echo Done