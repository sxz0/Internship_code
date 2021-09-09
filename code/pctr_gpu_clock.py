import time
import sys
from videocore.v3d import *
from videocore.driver import Driver

s=int(sys.argv[1])

with RegisterMapping(Driver()) as regmap:
        with PerformanceCounter(regmap, [13,14,15,16,17,18,19]) as pctr:
            time.sleep(s)
            result = pctr.result()
            print(sum(result))
