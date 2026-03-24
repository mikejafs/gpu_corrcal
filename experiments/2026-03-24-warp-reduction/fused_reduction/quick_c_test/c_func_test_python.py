import ctypes
import numpy as np

lib_path = "./libmyfuncs.so"
lib = ctypes.cdll.LoadLibrary(lib_path)

add = lib.add
add.argtypes = [ctypes.c_double, ctypes.c_double]
add.restype = ctypes.c_double
res = add(2, 3)

print(res)