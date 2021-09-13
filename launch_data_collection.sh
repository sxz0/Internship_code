#!/bin/bash

for ip in $(seq 15 25)
do
	echo "Execute './run_gpu_treasure_VC6.sh > feat_gpu_$ip &' in 192.168.0.$ip"
	sshpass -p "raspberry" ssh pi@192.168.0.$ip "cd py-videocore6; ./run_gpu_treasure_VC6.sh > feat_gpu_$ip &"
done

#for ip in $(seq 15 25)
#do
#	echo "Execute './run_gpu_treasure_VC4.sh > feat_gpu_$ip &' in 192.168.0.$ip"
#	sshpass -p "raspberry" ssh pi@192.168.0.$ip "cd py-videocore; ./run_gpu_treasure_VC4.sh > feat_gpu_$ip &"
#done
