#!/usr/bin/env bash
# Belge düzeyinde KL (doc_kld.cpp), GPU'da: llama.cpp aynı ağacın CUDA derlemesi (b12116, Kumru yaması dahil).
# Her model × dil ayrı süreç ve ayrı GPU; taban (bf16) ve bütün türler aynı süreçte sırayla.
# Kullanım: GPUS="0 1 2 3 4 5" [THREADS=16] [YENI=1] bash doc_kld_gpu.sh AD [AD ...]
# YENI=1: yalnız CSV'si olmayan türler.
set -euo pipefail
WORK=$HOME/work/tr-quant
LC=$WORK/llama.cpp
DOCS=$HOME/work/tr-serve/data/prompts/docs.json
THREADS=${THREADS:-16}
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5}"
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

if [ ! -x "$WORK/doc_kld_gpu" ] || [ "$WORK/doc_kld.cpp" -nt "$WORK/doc_kld_gpu" ]; then
  g++ -O3 -march=native -fopenmp -std=c++17 "$WORK/doc_kld.cpp" -I "$LC/include" -I "$LC/ggml/include" \
    -L "$LC/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LC/build/bin" -o "$WORK/doc_kld_gpu"
  log "doc_kld_gpu derlendi (llama.cpp $(git -C "$LC" rev-parse --short HEAD), CUDA)"
fi

i=0
for NAME in "$@"; do
  M=$WORK/models/$NAME
  R=$WORK/results/$NAME/belge
  mkdir -p "$R"
  "$WORK/.venv-convert/bin/python" "$WORK/doc_kld_prep.py" --hf "$M/hf" --docs "$DOCS" --out "$R"
  mapfile -t V < <(cd "$M" && ls ./*.gguf | sed 's|^\./||' | grep -v -e '^bf16\.gguf$' -e '^imatrix_')
  for lang in tr en; do
    W=()
    for v in "${V[@]}"; do
      [ "${YENI:-0}" = 1 ] && [ -s "$R/${lang}_${v%.gguf}.csv" ] && continue
      W+=("$v")
    done
    [ ${#W[@]} -eq 0 ] && { log "$NAME $lang: yeni tür yok"; continue; }
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    log "$NAME $lang: GPU $gpu, ${#W[@]} tür"
    CUDA_VISIBLE_DEVICES=$gpu DOC_KLD_NGL=99 "$WORK/doc_kld_gpu" "$M/bf16.gguf" "$R/tokens_$lang.bin" "$R/$lang" \
      "$THREADS" "${W[@]/#/$M/}" > "$R/doc_kld_gpu_$lang.log" 2>&1 &
    i=$((i + 1))
  done
done
wait
log "bitti"
