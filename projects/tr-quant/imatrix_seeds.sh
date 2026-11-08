#!/usr/bin/env bash
# Türkçe imatrix kazancının kalibrasyon örneğine duyarlılığı: tohum başına ayrı Vikipedi kalibrasyon kümesi
# (prepare_text.py --seed), TR ve EN imatrix, Q4_K_M, FLORES+ üzerinde TR ve EN KLD (flores_kld.sh'in tabanıyla).
# Bütün dizi tek shared kiralamayla çalışır.
# Kullanım: [SEEDS="1 2 3"] [GPUS=2,3,4,5] bash imatrix_seeds.sh AD [AD ...]
# Önkoşul: run_model.sh ve flores_kld.sh o model için bitmiş; $WORK/text/seed<N>/calib_{tr,en}.txt duruyor.
set -u
WORK=${WORK:-$HOME/work/tr-quant}
BIN=$WORK/llama.cpp/build/bin
SEEDS=${SEEDS:-"1 2 3"}
GPUS=${GPUS:-2,3,4,5}
IFS=, read -r -a GPU <<<"$GPUS"
PIN=(taskset -c 96-191)
CTX=512
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

id=$(~/bin/lease wait --owner tr-quant --gpus "$GPUS" --mode shared --hours 2 \
  --note "imatrix tohum duyarlılığı: $*, tohum $SEEDS" --timeout-min 240 | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "kiralama alınamadı"; exit 1; }
log "shared kiralama: $id (GPU $GPUS)"

# GPU'lara dağıtılan işler: her satır "komut"; aynı anda en fazla GPU sayısı kadar.
gpu_batch() {
  local slot=0 cmd
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    CUDA_VISIBLE_DEVICES=${GPU[$slot]} bash -c "$cmd" &
    slot=$(((slot + 1) % ${#GPU[@]}))
    [ "$slot" -eq 0 ] && wait
  done
  wait
}

imatrix_jobs() {
  for NAME in "$@"; do
    M=$WORK/models/$NAME R=$WORK/results/$NAME/flores-tohum
    mkdir -p "$R"
    for seed in $SEEDS; do
      for lang in tr en; do
        out=$M/imatrix_${lang}_s$seed.gguf
        [ -f "$out" ] && continue
        echo "${PIN[*]} $BIN/llama-imatrix -m $M/bf16.gguf -f $WORK/text/seed$seed/calib_$lang.txt -o $out" \
          "-c $CTX --chunks 200 -ngl 99 > $R/imatrix_${lang}_s$seed.log 2>&1"
      done
    done
  done
}
kld_jobs() {
  for NAME in "$@"; do
    M=$WORK/models/$NAME R=$WORK/results/$NAME/flores-tohum
    for seed in $SEEDS; do
      for imx in tr en; do
        for lang in tr en; do
          logf=$R/kld_Q4_K_M-$imx-s${seed}_$lang.log
          [ -s "$logf" ] && grep -q 'Mean    KLD' "$logf" && continue
          echo "${PIN[*]} $BIN/llama-perplexity -m $M/Q4_K_M-$imx-s$seed.gguf -c $CTX -ngl 99" \
            "--kl-divergence-base $M/kld_base_flores_$lang.bin --kl-divergence > $logf 2>&1"
        done
      done
    done
  done
}

log "imatrix"
imatrix_jobs "$@" | gpu_batch
log "quantize"
for NAME in "$@"; do
  M=$WORK/models/$NAME R=$WORK/results/$NAME/flores-tohum
  for seed in $SEEDS; do
    for imx in tr en; do
      out=$M/Q4_K_M-$imx-s$seed.gguf
      [ -f "$out" ] && continue
      "${PIN[@]}" "$BIN/llama-quantize" --imatrix "$M/imatrix_${imx}_s$seed.gguf" "$M/bf16.gguf" "$out" Q4_K_M \
        >"$R/quantize_Q4_K_M-$imx-s$seed.log" 2>&1 &
    done
  done
  wait
done
log "KLD"
kld_jobs "$@" | gpu_batch
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
