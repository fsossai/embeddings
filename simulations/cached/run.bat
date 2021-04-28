@echo off

set strategies=s_random.txt,s_hg.txt,s_knnp.txt,s_rr.txt,s_rc.txt,s_cr.txt,s_cc.txt

for %%s in (%strategies%) do (
    echo Strategy: %%s
    sim.exe                     ^
     --queries p.csv            ^
     --counts counts.txt        ^
     --n-processors 16          ^
     --min-size 100             ^
     --rel-size 0.01            ^
     --lookup-table %%s
    rem --sharding-name "name"
    echo --------------------
)