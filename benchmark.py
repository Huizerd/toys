import time
import random
import statistics

from tasks import fov_1d_speed_bounce, fov_1d_speed_wrap


functions = fov_1d_speed_bounce, fov_1d_speed_wrap
times = {f.__name__: [] for f in functions}
runs = 2000

for i in range(runs):
    func = random.choice(functions)
    t0 = time.time()
    func(30, 100, size=100)
    t1 = time.time()
    times[func.__name__].append((t1 - t0) * 1000)

for name, numbers in times.items():
    print(f"FUNCTION: {name} used {len(numbers)} times")
    print(f"\tMEDIAN {statistics.median(numbers)}")
    print(f"\tMEAN {statistics.mean(numbers)}")
    print(f"\tSTDEV {statistics.stdev(numbers)}")
