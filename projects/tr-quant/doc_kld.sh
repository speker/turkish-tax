#!/usr/bin/env bash
# Belge düzeyinde KL: servis ölçümünün 562 FLORES+ belgesi iki dilde, bütün token'lar puanlanarak
# (doc_kld.cpp). llama.cpp aynı ağaç (b12116, Kumru yaması dahil), CPU derlemesi; GPU sürümü doc_kld_gpu.sh.
# Kullanım: [THREADS=48] [CPUSETS="96-143 144-191"] bash doc_kld.sh AD [AD ...]
set -euo pipefail
WORK=$HOME/work/tr-quant
LC=$WORK/llama.cpp
DOCS=$HOME/work/tr-serve/data/prompts/docs.json
THREADS=${THREADS:-48}
read -r -a CPUSETS <<<"${CPUSETS:-96-143 144-191}"
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

if [ ! -x "$WORK/doc_kld" ] || [ "$WORK/doc_kld.cpp" -nt "$WORK/doc_kld" ]; then
  cmake -S "$LC" -B "$LC/build-cpu" -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$LC/build-cpu" -j 64 --target llama >/dev/null
  g++ -O3 -march=native -fopenmp -std=c++17 "$WORK/doc_kld.cpp" -I "$LC/include" -I "$LC/ggml/include" \
    -L "$LC/build-cpu/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LC/build-cpu/bin" -o "$WORK/doc_kld"
  log "doc_kld derlendi (llama.cpp $(git -C "$LC" rev-parse --short HEAD))"
fi

for NAME in "$@"; do
  M=$WORK/models/$NAME
  R=$WORK/results/$NAME/belge
  mkdir -p "$R"
  "$WORK/.venv-convert/bin/python" "$WORK/doc_kld_prep.py" --hf "$M/hf" --docs "$DOCS" --out "$R"
  mapfile -t V < <(cd "$M" && ls ./*.gguf | sed 's|^\./||' | grep -v -e '^bf16\.gguf$' -e '^imatrix_')
  log "$NAME: ${#V[@]} varyant"
  i=0
  for lang in tr en; do
    taskset -c "${CPUSETS[$i]}" "$WORK/doc_kld" "$M/bf16.gguf" "$R/tokens_$lang.bin" "$R/$lang" "$THREADS" \
      "${V[@]/#/$M/}" >"$R/doc_kld_$lang.log" 2>&1 &
    i=$((i + 1))
  done
  wait
  log "bitti: $R"
done
