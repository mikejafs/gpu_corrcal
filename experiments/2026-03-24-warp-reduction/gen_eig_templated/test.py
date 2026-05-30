import cupy as cp

print(cp.cuda.runtime.getDeviceProperties(0)['name'].decode())