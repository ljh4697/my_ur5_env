#!/bin/sh

### random
for i in $(seq 1 4); do
    python3 run_experiment.py -a batch_active_PBL -bm random -t tosser -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm random -t tosser -s 5 


for i in $(seq 6 9); do
    python3 run_experiment.py -a batch_active_PBL -bm random -t tosser -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm random -t tosser -s 10 



### greedy
for i in $(seq 1 4); do
    python3 run_experiment.py -a batch_active_PBL -bm greedy -t tosser -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm greedy -t tosser -s 5 


for i in $(seq 6 9); do
    python3 run_experiment.py -a batch_active_PBL -bm greedy -t tosser -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm greedy -t tosser -s 10 


### medoids
for i in $(seq 1 4); do
    python3 run_experiment.py -a batch_active_PBL -bm medoids -t tosser -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm medoids -t tosser -s 5 


for i in $(seq 6 9); do
    python3 run_experiment.py -a batch_active_PBL -bm medoids -t tosser -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm medoids -t tosser -s 10 


### dpp
for i in $(seq 1 4); do
    python3 run_experiment.py -a batch_active_PBL -bm dpp -t tosser -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm dpp -t tosser -s 5 


for i in $(seq 6 9); do
    python3 run_experiment.py -a batch_active_PBL -bm dpp -t tosser -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm dpp -t tosser -s 10 

## DPB
#sh DPB_grid_search.sh
