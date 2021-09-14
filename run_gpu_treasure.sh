#!/bin/bash
#for f in performance powersave
#do
#sudo cpufreq-set -g $f

rm feat_gpu_*
mac=$( cat /sys/class/net/eth0/address | tr : _ )
model=$( cat /proc/device-tree/model | sed 's/\x0//g' )
loop1=19
loop2=9
core=3
secs=240

if [[ $model == *"Pi 4"* ]];
then
	for s in `seq 0 $loop1` #39
	do
		for i in `seq 0 $loop2`
		do
			#taskset -c $i sudo PYTHONPATH=sandbox/ python3 examples/TREASURE_tests.py $i $f
			taskset -c $core sudo PYTHONPATH=sandbox/ python3 TREASURE_tests_VC6.py $secs >> feat_gpu_$mac
		done
		sleep 2
	done
elif [[ $model == *"Pi 3"* ]];
then
	for s in `seq 0 $loop1` #39
	do
		for i in `seq 0 $loop2`
		do
			#taskset -c $i sudo PYTHONPATH=sandbox/ python3 examples/TREASURE_tests.py $i $f
			taskset -c $core sudo python3 TREASURE_tests_VC4.py $secs >> feat_gpu_$mac
		done
		sleep 2
	done
else
	for s in `seq 0 $loop1` #39
	do
		for i in `seq 0 $loop2`
		do
			#taskset -c $i sudo PYTHONPATH=sandbox/ python3 examples/TREASURE_tests.py $i $f
			sudo nice --20 sudo python3 TREASURE_tests_VC4.py $secs >> feat_gpu_$mac
		done
		sleep 2
	done
fi

#done

