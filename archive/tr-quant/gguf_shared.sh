#!/usr/bin/env bash
# GGUF kuyruğunu (queue_gguf.sh) shared kiralamayla GPU 4-5'te çalıştırır. Önce tr-serve sıralayıcısının bitmesini bekler
# ve onun quiet kiralamasını bırakır (QUIET_ID), sonra shared kiralama bekleyip alır.
set -u
QUIET_ID=${QUIET_ID:-}
GPUS=${GPUS:-4,5}
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

if [ -n "$QUIET_ID" ]; then
  until grep -q '\] bitti' "$HOME/work/tr-serve/orchestrate.log" 2>/dev/null; do sleep 30; done
  ~/bin/lease release "$QUIET_ID" && log "quiet kiralama bırakıldı: $QUIET_ID"
fi

id=$(~/bin/lease wait --owner tr-quant --gpus "$GPUS" --mode shared --hours 6 \
  --note 'quant: Kumru-2B, Trendyol-LLM-8B-T1' --timeout-min 600 | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "kiralama alınamadı"; exit 1; }
log "shared kiralama: $id (GPU $GPUS)"
GPU_LIST=$GPUS CPUSET=96-191 bash "$HOME/work/tr-quant/queue_gguf.sh"
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
