#!/usr/bin/env bash
# Kumru-2B için uzun ölçüm. 2B model H200'de CPU sınırında; standart istem sayılarıyla koşular 15-20 sn
# sürüyor ve koşular arası fark %10-30. İstem sayısı dört kat, her yapılandırma 3 koşu.
# Ana sıralayıcı bitmeden başlamaz (GPU paylaşılmaz).
set -u
WORK=$HOME/work/tr-serve
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
until grep -q '\] bitti' "$WORK/orchestrate.log" 2>/dev/null; do sleep 60; done
for name in kumru-2b-uzun kumru-2b-uzun-fp8; do
  extra=()
  [[ "$name" == *-fp8 ]] && extra=(--quantization fp8)
  log "başla $name"
  if CONC="1:128 16:1024 64:4096 256:16384" REPS="1 2 3" \
     bash "$WORK/bench.sh" "$name" "$HOME/models/kumru-2b" 1 0 "${extra[@]}" >"$WORK/bench_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
done
log "bitti"
