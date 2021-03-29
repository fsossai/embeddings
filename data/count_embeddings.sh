#!/bin/bash
timestamp=$(date +%s)
python3 count_embeddings.py                     \
    -f /data/students/sossai/criteo/day_23.gz   \
    --gzip                                      \
    --chunk-size 10000000                       \
    --column-selection 14-39                    \
    --linear-plot                               \
    --highlight 1                               \
    --out-name counts_{timestamp}.png           \
    > count_${timestamp}.txt &