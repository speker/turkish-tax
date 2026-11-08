#!/usr/bin/env bash
# gpt-oss-120b, v3 ayarlarıyla (prefix caching kapalı), tam ısınma turundan sonra 2 ölçüm turu; iki ayrı sunucu
# paralel (GPU 2 ve 3). GPU 4'te kontrol: Turkish-Gemma BF16, v3'teki tek başına ölçümle (turkish-gemma-9b-t1-tek)
# karşılaştırılır; GPU 0'da ve CPU 0-95'te başka bir iş çalışırken etkilenmeyi gösterir. Hatlar CPU 96-191'de.
# Kiralama shared: makine bu sırada boş değil.
set -u
WORK=$HOME/work/tr-serve
export RESULTS_ROOT=$WORK/results-v3
TG=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
id=$(~/bin/lease wait --owner tr-serve --gpus 2,3,4 --mode shared --hours 2 --timeout-min 30 \
  --note 'gpt-oss ısınmalı 2 sunucu + Turkish-Gemma kontrol' | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "kiralama alınamadı"; exit 1; }
log "shared kiralama: $id (GPU 2,3,4)"
run() {
  local name=$1 dir=$2 gpu=$3 cpus=$4 port=$5
  shift 5
  log "başla $name (GPU $gpu, CPU $cpus)"
  if env "$@" CPUSET=$cpus PORT=$port bash "$WORK/bench.sh" "$name" "$dir" 1 "$gpu" --no-enable-prefix-caching \
    >"$WORK/bench_v3_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
}
run gpt-oss-120b-sicak-a "$HOME/models/gpt-oss-120b" 2 96-127 8102 WARMUP_FULL=1 &
run gpt-oss-120b-sicak-b "$HOME/models/gpt-oss-120b" 3 128-159 8103 WARMUP_FULL=1 &
run turkish-gemma-9b-t1-kontrol2 "$TG" 4 160-191 8104 &
wait
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
