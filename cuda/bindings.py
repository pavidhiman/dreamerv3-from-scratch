import ctypes
import numpy as np
import os

_lib = None
def _get_lib():
    global _lib 
    if _lib is not None:
        return _lib
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(lib_dir, 'matmul.so')
    if not os.path.exists(lib_path):
        raise RuntimeError(
            f'CUDA library not found at {lib_path}. '
            f'Compile with: nvcc -shared -o matmul.so matmul.cu'
        )
    _lib = ctypes.CDLL(lib_path)
    _lib.cuda_matmul.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.cuda_matmul.restype = None
    return _lib

def cuda_matmul(a, b): # works just like a @ b in numpy
    assert a.ndim == 2 and b.ndim == 2, 'inputs must be 2D'
    assert a.shape[1] == b.shape[0], f'shape mismatch: {a.shape} @ {b.shape}'

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    M, K = a.shape
    N = b.shape[1]
    c = np.empty((M, N), dtype=np.float32)

    lib = _get_lib()
    lib.cuda_matmul(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, K, N,
    )
    return c
