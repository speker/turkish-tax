#!/usr/bin/env bash
# FP8 kalite ölçümünün gürültü tabanı: aynı BF16 modeli ikinci kez koşup BF16'ya karşı BF16 karşılaştırması
# (fp8_quality.py compare/paired, etiket bf16b). Turkish-Gemma GPU 1, Ministral BF16 GPU 2; shared kiralama.
set -u
WORK=$HOME/work/tr-serve
IMAGE=tr-serve-vllm:0.33.0-bench
TEXT=$HOME/work/tr-quant/text
OUT=$WORK/results/fp8-kalite
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
id=$(~/bin/lease wait --owner tr-serve --gpus 1,2 --mode shared --hours 1 --timeout-min 30 \
  --note 'FP8 kalite gürültü tabanı (BF16 ikinci koşu)' | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "kiralama alınamadı"; exit 1; }
log "shared kiralama: $id"
null_run() {
  local name=$1 model=$2 gpu=$3 cpus=$4
  shift 4
  sudo docker run --rm --name "tr-serve-fp8null-$name" --runtime=nvidia --gpus "\"device=$gpu\"" --ipc=host \
    --cpuset-cpus "$cpus" -v "$model:/model:ro" -v "$TEXT:/text:ro" -v "$OUT/$name:/out" -v "$WORK:/work:ro" \
    --entrypoint python3 "$IMAGE" /work/fp8_quality.py run --model /model --label bf16b --text-dir /text --out /out "$@" \
    >"$OUT/$name/run_bf16b.log" 2>&1
  # paired burada tanımsız: iki koşu aynıysa içerik başına ek kayıp sıfır, oranı yok
  sudo docker run --rm --entrypoint python3 -v "$OUT/$name:/out" -v "$WORK:/work:ro" "$IMAGE" \
    /work/fp8_quality.py compare --dir /out --base bf16 --other bf16b >"$OUT/$name/compare_bf16b.log" 2>&1
  log "tamam $name"
}
null_run turkish-gemma-9b-t1 "$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf" 1 0-31 &
null_run ministral-3-14b-bf16 "$HOME/models/ministral-3-14b-bf16" 2 32-63 &
wait
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
