#!/bin/bash
#for f in performance powersave
#do
#sudo cpufreq-set -g $f
rm feat_gpu_*
mac=$( cat /sys/class/net/eth0/address | tr : _ )
loop1=19
loop2=9
core=3

for s in `seq 0 $loop1` #39
do
	for i in `seq 0 $loop2`
	do
		#taskset -c $i sudo PYTHONPATH=sandbox/ python3 examples/TREASURE_tests.py $i $f
		taskset -c $core sudo PYTHONPATH=sandbox/ python3 examples/TREASURE_tests_VC6.py 240 >> feat_gpu_$mac
	done
	sleep 2
done
#done
