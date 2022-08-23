#!/bin/sh

### random
for i in $(seq 1 4); do
    python3 run_experiment.py -a batch_active_PBL -bm random -t driver -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm random -t driver -s 5 


for i in $(seq 6 9); do
    python3 run_experiment.py -a batch_active_PBL -bm random -t driver -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm random -t driver -s 10 



### greedy
for i in $(seq 1 4); do
    python3 run_experiment.py -a batch_active_PBL -bm greedy -t driver -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm greedy -t driver -s 5 


for i in $(seq 6 9); do
    python3 run_experiment.py -a batch_active_PBL -bm greedy -t driver -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm greedy -t driver -s 10 


### medoids
for i in $(seq 1 4); do
    python3 run_experiment.py -a batch_active_PBL -bm medoids -t driver -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm medoids -t driver -s 5 


for i in $(seq 6 9); do
    python3 run_experiment.py -a batch_active_PBL -bm medoids -t driver -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm medoids -t driver -s 10 


### dpp
for i in $(seq 1 4); do
    python3 run_experiment.py -a batch_active_PBL -bm dpp -t driver -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm dpp -t driver -s 5 


for i in $(seq 6 9); do
    python3 run_experiment.py -a batch_active_PBL -bm dpp -t driver -s $i &
done

python3 run_experiment.py -a batch_active_PBL -bm dpp -t driver -s 10 