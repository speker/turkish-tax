#!/usr/bin/env bash
# Belebele'de GGUF türlerinin görev doğruluğu (belebele_gguf.cpp), GPU'da; llama.cpp aynı ağacın CUDA derlemesi.
# Her model × dil ayrı kuyruk; biten türler atlanır. Sonunda belebele_gguf_ozet.py.
# Kullanım: GPUS="0 1" [LEASE_ID=<bu oturumun etkin kiralaması>] [TURLER="bf16 Q4_K_M-tr ..."] bash belebele_gguf.sh AD [AD ...]
set -euo pipefail
WORK=$HOME/work/tr-quant
LC=$WORK/llama.cpp
DATA=$HOME/work/tr-serve/data/belebele
BBPY=$HOME/work/tr-serve/belebele_fp8.py
THREADS=${THREADS:-16}
TURLER=${TURLER:-"bf16 Q8_0 Q4_K_M-none Q4_K_M-tr Q4_K_M-en IQ4_XS-tr IQ4_XS-en IQ3_M-tr IQ3_M-en"}
read -r -a GPUS <<< "${GPUS:-0 1}"
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

id=${LEASE_ID:-}
if [ -z "$id" ]; then
  id=$(~/bin/lease wait --owner tr-quant --gpus "$(IFS=,; echo "${GPUS[*]}")" --mode shared --hours 2 \
    --note "tr-quant: GGUF Belebele dogrulugu: $*" --timeout-min 120 | awk 'NR==1 {print $1}')
  [ -n "$id" ] || { log "kiralama alınamadı"; exit 1; }
  own=1
fi
log "kiralama: $id (GPU ${GPUS[*]})"

if [ ! -x "$WORK/belebele_gguf" ] || [ "$WORK/belebele_gguf.cpp" -nt "$WORK/belebele_gguf" ]; then
  g++ -O3 -march=native -std=c++17 "$WORK/belebele_gguf.cpp" -I "$LC/include" -I "$LC/ggml/include" \
    -L "$LC/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LC/build/bin" -o "$WORK/belebele_gguf"
  log "belebele_gguf derlendi (llama.cpp $(git -C "$LC" rev-parse --short HEAD), CUDA)"
fi

queue() {
  local m=$1 r=$2 lang=$3 gpu=$4 t
  for t in $TURLER; do
    [ -f "$m/$t.gguf" ] || { log "yok: $m/$t.gguf"; continue; }
    [ -s "$r/${lang}_$t.csv" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu DOC_KLD_NGL=99 "$WORK/belebele_gguf" "$m/$t.gguf" "$r/sorular_$lang.bin" \
      "$r/${lang}_$t.csv.tmp" "$THREADS" >> "$r/belebele_gguf_$lang.log" 2>&1 &&
      mv "$r/${lang}_$t.csv.tmp" "$r/${lang}_$t.csv" && log "tamam $(basename "$m") $lang $t" ||
      log "HATA $(basename "$m") $lang $t"
  done
}

i=0
for NAME in "$@"; do
  M=$WORK/models/$NAME
  R=$WORK/results/$NAME/belebele
  mkdir -p "$R"
  "$WORK/.venv-convert/bin/python" "$WORK/belebele_gguf_prep.py" --hf "$M/hf" --data "$DATA" \
    --belebele-py "$BBPY" --out "$R"
  for lang in tr en; do
    queue "$M" "$R" "$lang" "${GPUS[$((i % ${#GPUS[@]}))]}" &
    i=$((i + 1))
  done
done
wait
for NAME in "$@"; do
  "$WORK/.venv-convert/bin/python" "$WORK/belebele_gguf_ozet.py" "$WORK/results/$NAME/belebele" --boot 10000
done
[ -n "${own:-}" ] && ~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
