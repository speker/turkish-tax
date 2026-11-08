#!/usr/bin/env bash
# Sistemin varsayılan CUDA'sını 12.8'e sabitler ve llama.cpp'yi CUDA 12.8 kütüphaneleriyle yeniden derler.
set -euo pipefail
WORK=${WORK:-$HOME/work/tr-quant}
sudo update-alternatives --set cuda /usr/local/cuda-12.8
echo "/usr/local/cuda -> $(readlink -f /usr/local/cuda)"
cd "$WORK/llama.cpp"
rm -rf build
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DLLAMA_CURL=OFF \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" --target llama-imatrix llama-quantize llama-perplexity >/dev/null 2>&1
ldd build/bin/libggml-cuda.so | grep -E 'cublas|cudart' | sed 's/ (0x.*//' | sort -u
