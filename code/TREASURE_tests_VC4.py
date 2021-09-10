import time
import os
import sys
import socket,fcntl,struct
from videocore.v3d import *
from videocore.driver import Driver

s=int(sys.argv[1])
def get_gpu_cycles(s):
    with RegisterMapping(Driver()) as regmap:
        with PerformanceCounter(regmap, [13,14,15,16,17,18,19]) as pctr:
            time.sleep(s)
            result = pctr.result()
            return sum(result)

def getHwAddr(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    info = fcntl.ioctl(s.fileno(), 0x8927,  struct.pack('256s', bytes(ifname, 'utf-8')[:15]))
    return ':'.join('%02x' % b for b in info[18:24])

def main():
	results=[]
	results.append(os.popen("vcgencmd measure_temp | cut -d = -f 2 | cut -d \"'\" -f 1").read()[:-1])
	results.append(get_gpu_cycles(s))
	
	results.append(getHwAddr('eth0'))
	print(*results, sep=',')

if __name__ == "__main__":
    main()
