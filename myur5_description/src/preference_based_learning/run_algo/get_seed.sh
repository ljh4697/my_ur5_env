#!/bin/sh


for i in $(seq 1 10); do
    python3 run_experiment.py -t tosser -d 0.7 -w 0.006 -g 0.94 -l 0.9 -s $i
done


# for i in $(seq 1 10); do
#     python3 run_experiment.py -t tosser -d 0.7 -w 0.006 -g 0.947 -l 0.10 -s $i
# done

