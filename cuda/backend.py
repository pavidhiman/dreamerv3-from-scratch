import numpy as np
import ctypes
import os

USE_CUDA = False
_lib = None


def _try_load_cuda():
    global USE_CUDA, _lib
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(lib_dir, 'matmul_tiled.so')
    if not os.path.exists(lib_path):
        lib_path = os.path.join(lib_dir, 'matmul.so')
    if not os.path.exists(lib_path):
        return
    try:
        _lib = ctypes.CDLL(lib_path)
        func_name = 'cuda_matmul_tiled' if 'tiled' in lib_path else 'cuda_matmul'
        func = getattr(_lib, func_name)
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        func.restype = None
        USE_CUDA = True
        print(f'[backend] CUDA loaded: {os.path.basename(lib_path)}')
    except Exception as e:
        print(f'[backend] CUDA load failed: {e}, using NumPy')


_try_load_cuda()


def matmul(a, b):
    if USE_CUDA and a.ndim == 2 and b.ndim == 2:
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        M, K = a.shape
        N = b.shape[1]
        c = np.empty((M, N), dtype=np.float32)
        func_name = 'cuda_matmul_tiled' if 'tiled' in _lib._name else 'cuda_matmul'
        func = getattr(_lib, func_name)
        func(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, K, N,
        )
        return c
    return np.matmul(a, b)
