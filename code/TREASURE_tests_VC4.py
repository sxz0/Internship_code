import time
import os
import sys
import socket,fcntl,struct
from videocore.v3d import *
from videocore.driver import Driver
import random
import hashlib


def sleep(duration, get_now=time.perf_counter):
    now = time.time_ns()
    duration=duration*1000000000
    end = now + duration
    while now < end:
        now = time.time_ns()

def get_QPU_freq(s):
    with RegisterMapping(Driver()) as regmap:
        with PerformanceCounter(regmap, [13,14,15,16,17,18,19]) as pctr:
            time.sleep(s)
            result = pctr.result()
            return sum(result)

def cpu_true_random(n):
     with RegisterMapping(Driver()) as regmap:
        with PerformanceCounter(regmap, [13,14,15,16,17,28,19]) as pctr:
            a=os.urandom(n)
            result = pctr.result()
            return (sum(result))

def cpu_hash():
    with RegisterMapping(Driver()) as regmap:
        with PerformanceCounter(regmap, [13,14,15,16,17,28,19]) as pctr:
            h=int(hashlib.sha256("test string".encode('utf-8')).hexdigest(), 16) % 10**8
            result = pctr.result()
            return (sum(result))

def getHwAddr(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    info = fcntl.ioctl(s.fileno(), 0x8927,  struct.pack('256s', bytes(ifname, 'utf-8')[:15]))
    return ':'.join('%02x' % b for b in info[18:24])

def main():
	s=int(sys.argv[1])
	results=[]
	results.append(os.popen("vcgencmd measure_temp | cut -d = -f 2 | cut -d \"'\" -f 1").read()[:-1])
	results.append(get_QPU_freq(s))
	
	results.append(getHwAddr('eth0'))
	print(*results, sep=',')

if __name__ == "__main__":
    main()

