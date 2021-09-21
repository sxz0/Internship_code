#!/bin/bash

rm feat_gpu_*

if [ -e /sys/class/net/eth0 ]; then
	mac=$( cat /sys/class/net/eth0/address | tr : _ )
elif [ -e /sys/class/net/enx* ]; then
	mac=$( cat /sys/class/net/enx*/address | tr : _ )
elif [ -e /sys/class/net/wlan0 ]; then
	mac=$( cat /sys/class/net/wlan0/address | tr : _ )
else
	mac=$( cat /sys/class/net/wlx*/address | tr : _ )
fi

model=$( cat /proc/device-tree/model | sed 's/\x0//g' )
loop1=19
loop2=9
core=3
secs=600
random=1000000

if [[ $model == *"Pi 4"* ]];
then
	for s in `seq 0 $loop1` #39
	do
		for i in `seq 0 $loop2`
		do
			taskset -c $core sudo nice --20 sudo PYTHONPATH=sandbox/ python3 TREASURE_tests_VC6.py $secs $random >> feat_gpu_$mac
		done
		sleep 2
	done
elif [[ $model == *"Pi 3"* ]];
then
	for s in `seq 0 $loop1` #39
	do
		for i in `seq 0 $loop2`
		do
			taskset -c $core sudo nice --20 sudo python3 TREASURE_tests_VC4.py $secs $random >> feat_gpu_$mac
		done
		sleep 2
	done
else
	for s in `seq 0 $loop1` #39
	do
		for i in `seq 0 $loop2`
		do
			sudo nice --20 sudo python3 TREASURE_tests_VC4.py $secs $random >> feat_gpu_$mac
		done
		sleep 2
	done
fi

#done

