#!/bin/bash
#for f in performance powersave
#do
#sudo cpufreq-set -g $f
rm feat_gpu_*
mac=$( cat /sys/class/net/eth0/address | tr : _ )
for s in `seq 0 19` #39
do
	for i in `seq 0 9`
	do
		#taskset -c $i sudo PYTHONPATH=sandbox/ python3 examples/TREASURE_tests.py $i $f
		taskset -c 3 sudo PYTHONPATH=sandbox/ python3 examples/TREASURE_tests_VC6.py 240 >> feat_gpu_$mac
	done
	sleep 2
done
#done
