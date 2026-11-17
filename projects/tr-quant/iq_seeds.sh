#!/usr/bin/env bash
# IQ3_M ve IQ4_XS'te kalibrasyon dili, dört kalibrasyon örneğiyle (Q4_K_M'deki imatrix_seeds.sh'ın tohumları):
# eksik IQ türleri CPU'da nicemlenir, sonra yalnız yeni türler için belge düzeyinde KL (doc_kld_gpu.sh YENI=1).
# Bütün dizi tek shared kiralamayla. Kullanım: [GPUS=0,1] bash iq_seeds.sh AD [AD ...]
set -u
WORK=$HOME/work/tr-quant
BIN=$WORK/llama.cpp/build/bin
GPUS=${GPUS:-0,1}
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

id=$(~/bin/lease wait --owner tr-quant --gpus "$GPUS" --mode shared --hours 2 \
  --note "tr-quant: IQ3_M/IQ4_XS kalibrasyon tohumlari ve belge duzeyinde KL: $*" --timeout-min 120 | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "kiralama alınamadı"; exit 1; }
log "shared kiralama: $id (GPU $GPUS)"

log "quantize"
cpu=96
for NAME in "$@"; do
  M=$WORK/models/$NAME R=$WORK/results/$NAME/belge
  mkdir -p "$R"
  for type in IQ3_M IQ4_XS; do
    for seed in 1 2 3; do
      for imx in tr en; do
        out=$M/$type-$imx-s$seed.gguf
        [ -f "$out" ] && continue
        taskset -c "$cpu-$((cpu + 7))" "$BIN/llama-quantize" --imatrix "$M/imatrix_${imx}_s$seed.gguf" \
          "$M/bf16.gguf" "$out.tmp" "$type" 8 > "$R/quantize_$type-$imx-s$seed.log" 2>&1 &&
          mv "$out.tmp" "$out" &
        cpu=$(((cpu + 8 - 96) % 96 + 96))
      done
    done
  done
done
wait
for NAME in "$@"; do
  n=$(ls "$WORK/models/$NAME"/IQ*-s?.gguf 2> /dev/null | wc -l)
  log "$NAME: $n/12 yeni IQ türü"
done

log "KL"
GPUS="${GPUS//,/ }" YENI=1 bash "$WORK/doc_kld_gpu.sh" "$@"
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
