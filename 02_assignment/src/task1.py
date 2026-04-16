import cupy as cp

'''
CuPy exposes all CUDA device attributes through a single dictionary.
cp.cuda.Device().attributes.items()
'''
# return dictionary of attribute values with the names as keys
attr = dict(cp.cuda.Device().attributes.items())

print("L2CacheSize:", attr["L2CacheSize"])
print("MaxSharedMemoryPerMultiprocessor:", attr["MaxSharedMemoryPerMultiprocessor"])
print("ClockRate:", attr["ClockRate"])

print(cp.cuda.Device().attributes.items())