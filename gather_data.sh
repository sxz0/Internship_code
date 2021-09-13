#!/bin/bash

# Data from RPi4
for ip in $(seq 15 25)
do
	sshpass -p "raspberry" scp pi@192.168.0.$ip:/home/pi/py-videocore6/feat_gpu_* ./datasets/
done



# Data from RPi3
for ip in $(seq 4 14)
do
	sshpass -p "raspberry" scp pi@192.168.0.$ip:/home/pi/py-videocore/feat_gpu_* ./datasets/
done

cat ./datasets/feat_gpu_* >> ./datasets/current_dataset.csv
