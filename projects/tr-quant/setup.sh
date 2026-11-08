#!/usr/bin/env bash
# h200'de tr-quant ortamı: llama.cpp (varsayılan b12116, CUDA) ve dönüştürme için Python ortamı.
# Kullanım: bash setup.sh [LLAMA_CPP_TAG]
set -euo pipefail
WORK=${WORK:-$HOME/work/tr-quant}
TAG=${1:-b12116}
mkdir -p "$WORK"
cd "$WORK"

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake ccache >/dev/null

if [ ! -d llama.cpp ]; then
  git clone -q https://github.com/ggml-org/llama.cpp
fi
cd llama.cpp
git fetch -q --tags
git checkout -q "$TAG"
if [ "$TAG" = b12116 ] && [ "$(git rev-parse HEAD)" != 5851a8e505f4d5fbb183f2847698eb6155d176b7 ]; then
  echo "b12116 beklenen commit değil: $(git rev-parse HEAD)" >&2
  exit 1
fi
echo "llama.cpp $TAG $(git rev-parse --short HEAD)"
# Sürücü 570 en fazla CUDA 12.8'i destekler; makinede CUDA 13.0 da kurulu ve /usr/local/cuda ona gidebilir.
# Derleyici ve kütüphaneler 12.8'e sabitlenir, yoksa cuBLAS çalışma anında başlatılamaz.
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DLLAMA_CURL=OFF \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" --target llama-imatrix llama-quantize llama-perplexity >/dev/null
cd "$WORK"

export PATH=$HOME/.local/bin:$PATH
# İki ortam: .venv indirme ve metin hazırlığı (huggingface_hub 2.x), .venv-convert llama.cpp dönüştürme
# (transformers 4.x huggingface_hub < 1.0 istiyor; aynı ortamda çakışırlar).
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python "huggingface_hub==2.4.0" "pyarrow==26.0.1"
uv venv -q --python 3.12 .venv-convert
uv pip install -q --python .venv-convert/bin/python \
  -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
  --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu

{
  echo "llama.cpp: $TAG $(git -C llama.cpp rev-parse HEAD)"
  echo "nvcc: $(/usr/local/cuda-12.8/bin/nvcc --version | awk '/release/ {print $6}')"
  echo "sürücü: $(cat /sys/module/nvidia/version)"
  echo "indirme ortamı:"
  uv pip freeze --python .venv/bin/python | grep -iE '^(huggingface.hub|pyarrow)=='
  echo "dönüştürme ortamı:"
  uv pip freeze --python .venv-convert/bin/python | grep -iE '^(gguf|torch|transformers|numpy|safetensors|sentencepiece|huggingface.hub)=='
} > "$WORK/ortam.txt"
cat "$WORK/ortam.txt"
