#!/bin/bash

for ip in $(seq 4 14)
do
	echo "Execute './run_gpu_treasure.sh > feat_gpu_$ip &' in 192.168.0.$ip"
	sshpass -p "raspberry" ssh pi@192.168.0.$ip "cd py-videocore6; ./run_gpu_treasure.sh > feat_gpu_$ip &"
done
