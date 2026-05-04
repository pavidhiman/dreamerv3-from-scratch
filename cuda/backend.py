import numpy as np
import ctypes
import os

USE_CUDA = False
_matmul_func = None
_gru_func = None


def _try_load_cuda():
    global USE_CUDA, _matmul_func, _gru_func
    lib_dir = os.path.dirname(os.path.abspath(__file__))

    matmul_path = os.path.join(lib_dir, 'matmul_tiled.so')
    if not os.path.exists(matmul_path):
        matmul_path = os.path.join(lib_dir, 'matmul.so')

    if os.path.exists(matmul_path):
        try:
            lib = ctypes.CDLL(matmul_path)
            name = 'cuda_matmul_tiled' if 'tiled' in matmul_path else 'cuda_matmul'
            _matmul_func = getattr(lib, name)
            _matmul_func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            _matmul_func.restype = None
            USE_CUDA = True
            print(f'[backend] matmul loaded: {os.path.basename(matmul_path)}')
        except Exception as e:
            print(f'[backend] matmul load failed: {e}')

    gru_path = os.path.join(lib_dir, 'gru.so')
    if os.path.exists(gru_path):
        try:
            lib = ctypes.CDLL(gru_path)
            _gru_func = lib.cuda_gru_forward
            _gru_func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            _gru_func.restype = None
            print('[backend] GRU loaded: gru.so')
        except Exception as e:
            print(f'[backend] GRU load failed: {e}')

    if not USE_CUDA:
        pass


_try_load_cuda()


def _ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def matmul(a, b):
    if _matmul_func is not None and a.ndim == 2 and b.ndim == 2:
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        M, K = a.shape
        N = b.shape[1]
        c = np.empty((M, N), dtype=np.float32)
        _matmul_func(
            _ptr(a), _ptr(b), _ptr(c),
            M, K, N,
        )
        return c
    return np.matmul(a, b)


def gru_forward(x, h_prev, W_r, b_r, W_z, b_z, W_c, b_c):
    if _gru_func is None:
        return None

    x = np.ascontiguousarray(x, dtype=np.float32)
    h_prev = np.ascontiguousarray(h_prev, dtype=np.float32)
    W_r = np.ascontiguousarray(W_r, dtype=np.float32)
    b_r = np.ascontiguousarray(b_r, dtype=np.float32)
    W_z = np.ascontiguousarray(W_z, dtype=np.float32)
    b_z = np.ascontiguousarray(b_z, dtype=np.float32)
    W_c = np.ascontiguousarray(W_c, dtype=np.float32)
    b_c = np.ascontiguousarray(b_c, dtype=np.float32)

    batch, input_size = x.shape
    hidden_size = h_prev.shape[1]
    h_out = np.empty((batch, hidden_size), dtype=np.float32)

    _gru_func(
        _ptr(x), _ptr(h_prev),
        _ptr(W_r), _ptr(b_r),
        _ptr(W_z), _ptr(b_z),
        _ptr(W_c), _ptr(b_c),
        _ptr(h_out),
        batch, input_size, hidden_size,
    )
    return h_out
