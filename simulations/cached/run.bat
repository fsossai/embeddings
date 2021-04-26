@echo off

sim.exe                     ^
 --queries p.csv            ^
 --counts counts.txt        ^
 --n-processors 16          ^
 --min-size 100             ^
 --rel-size 0.01            ^
 --lookup-table s_alpha1_100.txt
