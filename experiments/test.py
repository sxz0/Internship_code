import os
from random import shuffle
import tracemalloc
import pandas as pd
from time import time_ns as time
"""
try:  # if Python >= 3.3 use new high-res counter
    from time import perf_counter as time
except ImportError:  # else select highest available resolution counter
    if sys.platform[:3] == 'win':
        from time import clock as time
    else:
        from time import time
"""


def tracing_start():
    tracemalloc.stop()
    tracemalloc.start()


def tracing_mem():
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak / (1024 * 1024)
    return peak


def fib(n):
    if n <= 1: return 1
    return fib(n - 1) + fib(n - 2)


def write_test(file, block_size, blocks_count, show_progress=False):
    f = os.open(file, os.O_CREAT | os.O_WRONLY, 0o777)  # low-level I/O

    took = []
    for i in range(blocks_count):
        buff = os.urandom(block_size)
        start = time()
        os.write(f, buff)
        os.fsync(f)  # force write to disk
        t = time() - start
        took.append(t)

    os.close(f)
    return took


def read_test(file, block_size, blocks_count, show_progress=False):
    f = os.open(file, os.O_RDONLY, 0o777)  # low-level I/O
    # generate random read positions
    offsets = list(range(0, blocks_count * block_size, block_size))
    shuffle(offsets)

    took = []
    for i, offset in enumerate(offsets, 1):
        start = time()
        os.lseek(f, offset, os.SEEK_SET)  # set position
        buff = os.read(f, block_size)  # read from position
        t = time() - start
        if not buff: break  # if EOF reached
        took.append(t)

    os.close(f)
    return took


while (True):
    tracing_start()
    start = time()
    sq_list = []
    for elem in range(1, 1000):
        sq_list.append(elem + elem ** 2)
    # print(sq_list)
    end = time()
    peak1 = tracing_mem()
    t1 = end - start

    tracing_start()
    start = time()
    df = pd.read_csv("test_dataset.csv")
    end = time()
    peak2 = tracing_mem()
    t2 = end - start

    print(str(t1) + "," + str(t2) + "," +
          str(write_test("test", 10240, 100)) + "," +
          str(read_test("test", 10240, 100)))
