import sys
import time

f = open('test.txt', 'w')

sys.stdout = f

for i in range(5):
    time.sleep(i)
    s_t = time.time()
    print(i, flush=True)
    print(time.time() - s_t, flush=True)
f.close()