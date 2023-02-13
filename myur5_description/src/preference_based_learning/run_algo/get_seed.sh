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







for i in $(seq 1 10); do
    python3 run_experiment.py -w 0.25 -g 0.935 -l 0.8 -s $i
done


for i in $(seq 1 10); do
    python3 run_experiment.py -w 0.0004 -g 0.94 -l 0.8 -s $i
done


for i in $(seq 1 10); do
    python3 run_experiment.py -w 0.002 -g 0.935 -l 0.8 -s $i
done

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
