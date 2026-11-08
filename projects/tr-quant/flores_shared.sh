#!/usr/bin/env bash
# flores_kld.sh'i shared kiralamayla çalıştırır. Kullanım: [GPUS=2,3,4,5] bash flores_shared.sh AD [AD ...]
set -u
GPUS=${GPUS:-2,3,4,5}
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
id=$(~/bin/lease wait --owner tr-quant --gpus "$GPUS" --mode shared --hours 2 \
  --note "FLORES+ aynı içerik KLD: $*" --timeout-min 120 | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "kiralama alınamadı"; exit 1; }
log "shared kiralama: $id (GPU $GPUS)"
GPU_LIST=$GPUS CPUSET=96-191 bash "$HOME/work/tr-quant/flores_kld.sh" "$@" || log "HATA flores_kld"
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
