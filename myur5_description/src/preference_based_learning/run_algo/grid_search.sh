#!/bin/sh


for i in $(seq 1 3); do
    python3 run_experiment.py -s $i

done
