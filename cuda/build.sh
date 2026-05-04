#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Compiling matmul.cu -> matmul.so"
nvcc -shared -Xcompiler -fPIC -o matmul.so matmul.cu

echo "Done. Library at: $SCRIPT_DIR/matmul.so"
