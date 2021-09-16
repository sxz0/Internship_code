import time

def sleep(duration):
    duration=duration*1000000000
    now = time.perf_counter_ns()
    end = now + duration
    while now < end:
        now = time.perf_counter_ns()

while True:
    a=time.perf_counter_ns()
    sleep(0.0000001)
    print(time.perf_counter_ns()-a)

    a=time.perf_counter_ns()
    time.sleep(0.0000001)
    print(time.perf_counter_ns()-a)
    print("\n")