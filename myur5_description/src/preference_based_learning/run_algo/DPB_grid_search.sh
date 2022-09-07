#!/bin/sh




# for i in $(seq 1 4); do


#     for g in $(seq 1 8); do


#         for d in $(seq 6 7); do

#             for s in $(seq 1 4); do

#                 for t in $(seq 1 10); do

#                     z=$( echo "scale=5; 1/10^$i" | bc -l)
#                     x=$( echo "scale=5; (90+$g)/100" | bc -l)
#                     c=$( echo "scale=5; $d/10" | bc -l)
#                     v=$( echo "scale=5; 0.8+0.1*$s" | bc -l)


#                     python3 run_experiment.py -t lunarlander -w $z -g $x -d $c -l $v -s $t

#                 done

#             done

#         done

#     done

# done

# fine
for i in $(seq 1 7); do


    for g in $(seq 1 5); do

        #for d in $(seq 6 7); do


        for s in $(seq 1 4); do

        z=$( echo "scale=5; 0.0005*$i" | bc -l)
        x=$( echo "scale=5; 0.93+0.005*$g" | bc -l)
        c=$( echo "scale=5; $d/10" | bc -l)
        v=$( echo "scale=5; 0.1*2*$s" | bc -l)


        python3 run_experiment.py -t tosser -w $z -g $x -d 0.7 -l $v -s 5

        done

        #done


    done

done
