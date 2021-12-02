#!/bin/bash



for ip in $(seq 27 36)
do
	echo "Execute '$@' in 192.168.0.$ip"
	sshpass -p "raspberry" ssh pi@192.168.0.$ip "$@"
	echo ""

done


