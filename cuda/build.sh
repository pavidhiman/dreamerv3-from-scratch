#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Compiling matmul_tiled.cu -> matmul_tiled.so"
nvcc -shared -Xcompiler -fPIC -o matmul_tiled.so matmul_tiled.cu

echo "Compiling gru.cu -> gru.so"
nvcc -shared -Xcompiler -fPIC -o gru.so gru.cu

echo "Done. Libraries at: $SCRIPT_DIR/"
ls -la *.so
