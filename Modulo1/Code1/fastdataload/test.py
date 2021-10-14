from core.readfile import fastload
import time

# test data
emlFile = b"data/data.dat"
start = time.time()
data = fastload(emlFile, int(1e7+2000))
print(time.time()-start)
print(data[:20])
