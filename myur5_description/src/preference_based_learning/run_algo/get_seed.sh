#!/bin/sh




for i in $(seq 1 10); do
    python3 run_experiment.py -t tosser -d 0.7 -w 0.0008 -g 0.955 -l 0.2 -s $i
done ####

for i in $(seq 1 10); do
    python3 run_experiment.py -t tosser -d 0.7 -w 0.0008 -g 0.954 -l 0.2 -s $i
done ####

for i in $(seq 1 10); do
    python3 run_experiment.py -t tosser -d 0.7 -w 0.0009 -g 0.954 -l 0.21 -s $i
done ####

for i in $(seq 1 10); do
    python3 run_experiment.py -t tosser -d 0.7 -w 0.0009 -g 0.954 -l 0.19 -s $i
done ####

for i in $(seq 1 10); do
    python3 run_experiment.py -t tosser -d 0.7 -w 0.0008 -g 0.954 -l 0.22 -s $i
done ####

for i in $(seq 1 10); do
    python3 run_experiment.py -t tosser -d 0.7 -w 0.0008 -g 0.954 -l 0.18 -s $i
done ####

for i in $(seq 1 10); do
    python3 run_experiment.py -t tosser -d 0.7 -w 0.0009 -g 0.954 -l 0.22 -s $i
done ####

for i in $(seq 1 10); do
    python3 run_experiment.py -t tosser -d 0.7 -w 0.0009 -g 0.954 -l 0.18 -s $i
done ####

# for i in $(seq 1 10); do
#     python3 run_experiment.py -t avoid -d 0.7 -w 0.0005 -g 0.95 -l 0.44 -s $i
# done

# for i in $(seq 1 10); do
#     python3 run_experiment.py -t avoid -d 0.7 -w 0.0007 -g 0.95 -l 0.42 -s $i
# done

# for i in $(seq 1 10); do
#     python3 run_experiment.py -t avoid -d 0.7 -w 0.0005 -g 0.952 -l 0.44 -s $i
# done


# for i in $(seq 1 10); do
#     python3 run_experiment.py -t avoid -d 0.7 -w 0.0005 -g 0.955 -l 0.44 -s $i
# done

# for i in $(seq 1 10); do
#     python3 run_experiment.py -t avoid -d 0.6 -w 0.0002 -g 0.95 -l 0.42 -s $i
# done

# for i in $(seq 1 10); do
#     python3 run_experiment.py -t avoid -d 0.6 -w 0.0002 -g 0.951 -l 0.42 -s $i
# done
# for i in $(seq 1 10); do
#     python3 run_experiment.py -t tosser -d 0.7 -w 0.006 -g 0.947 -l 0.10 -s $i
# done

7