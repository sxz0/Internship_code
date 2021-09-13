#!/bin/bash

for ip in $(seq 15 25)
do
	sshpass -p "raspberry" scp pi@192.168.0.$ip:/home/pi/py-videocore6/feat_gpu_* ./datasets/
done

cat ./datasets/feat_gpu_* >> ./datasets/current_dataset.csv
