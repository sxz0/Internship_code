#!/bin/bash

ip=$1
sshpass -p "raspberry" ssh pi@192.168.0.$ip
