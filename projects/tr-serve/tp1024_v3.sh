#!/usr/bin/env bash
# Qwen3.8-27B TP2 ve TP4, 1024 eşzamanlılık, istemci dosya sınırı yükseltilmiş (bench.sh --ulimit). tp_high_v3.sh'teki
# 1024 koşularında istemci "Too many open files" ile isteklerin çoğunu düşürdü; o dosyalar gecersiz_ulimit/ altında.
# TP2 GPU 0,5 ve TP4 GPU 1-4 aynı anda.
set -u
WORK=$HOME/work/tr-serve
export RESULTS_ROOT=$WORK/results-v3
M=$HOME/models/qwen3.8-27b
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
id=$(~/bin/lease wait --owner tr-serve --gpus all --mode quiet --hours 2 --timeout-min 30 \
  --note 'Qwen3.8 TP2/TP4 1024 eşzamanlılık' | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "kiralama alınamadı"; exit 1; }
log "quiet kiralama: $id"
run() {
  local name=$1 tp=$2 gpus=$3 cpus=$4 port=$5
  log "başla $name (TP=$tp, GPU $gpus, CPU $cpus)"
  if CONC="1024:16384" CPUSET=$cpus PORT=$port bash "$WORK/bench.sh" "$name" "$M" "$tp" "$gpus" \
    --no-enable-prefix-caching >"$WORK/bench_v3_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
}
run qwen3.8-27b-tp2-1024 2 0,5 128-191 8102 &
run qwen3.8-27b-tp4-1024 4 1,2,3,4 0-127 8101 &
wait
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
