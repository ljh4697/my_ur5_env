#!/bin/sh


# python3 run_experiment.py -t tosser -d 0.7 -w 0.0006 -g 0.955 -l 0.1
# python3 run_experiment.py -t tosser -d 0.7 -w 0.0006 -g 0.953 -l 0.1





# for i in $(seq 1 5); do
#     python3 run_experiment.py -t tosser -d 0.7 -w 0.0009 -g 0.954 -l 0.1 -s $i
# done ####

# python3 run_experiment.py -t driver -d 0.7 -w 0.0002 -g 0.95 -l 0.1 
# python3 run_experiment.py -t driver -d 0.7 -w 0.0002 -g 0.948 -l 0.11
# python3 run_experiment.py -t driver -d 0.7 -w 0.0002 -g 0.951 -l 0.1 
# python3 run_experiment.py -t driver -d 0.7 -w 0.0002 -g 0.955 -l 0.11 
# python3 run_experiment.py -t driver -d 0.7 -w 0.0002 -g 0.956 -l 0.1
# python3 run_experiment.py -t driver -d 0.7 -w 0.0002 -g 0.947 -l 0.1 


# for i in $(seq 3 5); do
#     python3 run_experiment.py -t tosser -d 0.7 -w 0.0009 -g 0.954 -l 0.1 -s $i
# done ####

# for i in $(seq 2 5); do
#     python3 run_experiment.py -t tosser -d 0.7 -w 0.0008 -g 0.954 -l 0.11 -s $i
# done ####






# for i in $(seq 1 10); do
#     python3 run_experiment.py -a DPB -d 0.7 -w 0.0019 -g 0.934 -l 0.6 -s $i -t avoid &
# done

# for i in $(seq 1 10); do
#     python3 run_experiment.py -a batch_active_PBL -bm information -t driver -s $i
# done


for i in $(seq 1 10); do
    python3 run_experiment.py -a batch_active_PBL -bm information -t avoid -s $i
done


# for i in $(seq 1 10); do
#     python3 run_experiment.py -t avoid -w 0.0019 -g 0.934 -d 0.7 -l 0.6 -s $i -a DPB
# done

