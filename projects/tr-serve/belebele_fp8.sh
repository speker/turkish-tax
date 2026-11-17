#!/usr/bin/env bash
# FP8'in görev doğruluğuna etkisi: Belebele Türkçe ve İngilizce, BF16 ve FP8.
# Kullanım: bash belebele_fp8.sh "AD:GPU:CPU" ...   (AD: turkish-gemma-9b-t1 gemma-4-12b-it qwen3.8-27b kumru-2b
#           ministral-3-14b-bf16); veri ~/work/tr-serve/data/belebele/{tur_Latn,eng_Latn}.jsonl
set -u
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
WORK=$HOME/work/tr-serve
M=$HOME/models
IMAGE=tr-serve-vllm:0.33.0-bench
OUT=$WORK/results/belebele-fp8
declare -A PATH_OF=(
  [turkish-gemma-9b-t1]=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf [gemma-4-12b-it]=$M/gemma-4-12b-it
  [qwen3.8-27b]=$M/qwen3.8-27b [kumru-2b]=$M/kumru-2b [ministral-3-14b-bf16]=$M/ministral-3-14b-bf16
)
brun() {
  local name=$1 label=$2 gpu=$3 cpus=$4
  shift 4
  mkdir -p "$OUT/$name"
  [ -f "$OUT/$name/${label}_en.npz" ] && return 0
  sudo docker run --rm --name "tr-serve-bb-$name-$label" --runtime=nvidia --gpus "\"device=$gpu\"" --ipc=host \
    --cpuset-cpus "$cpus" -v "${PATH_OF[$name]}:/model:ro" -v "$OUT/$name:/out" -v "$WORK:/work:ro" \
    --entrypoint python3 "$IMAGE" /work/belebele_fp8.py run --model /model --label "$label" \
    --data /work/data/belebele --out /out "$@" > "$OUT/$name/run_$label.log" 2>&1
  [ -f "$OUT/$name/${label}_en.npz" ] && log "tamam belebele $name $label: $(grep -h 'doğruluk' "$OUT/$name/run_$label.log" | tr '\n' ' ')" ||
    log "HATA belebele $name $label"
}
one() {
  local name=$1 gpu=$2 cpus=$3 extra=() s=""
  # CHAT=1: sohbet şablonu, düşünme kapalı (etiketler -chat); şablon BOS'u kendisi koyar
  if [ "${CHAT:-0}" = 1 ]; then
    extra=(--chat)
    s=-chat
  elif [ "$name" = gemma-4-12b-it ]; then
    extra=(--add-bos)
  fi
  brun "$name" "bf16$s" "$gpu" "$cpus" "${extra[@]}"
  brun "$name" "fp8$s" "$gpu" "$cpus" --quantization fp8 "${extra[@]}"
  sudo docker run --rm --entrypoint python3 -v "$OUT/$name:/out" -v "$WORK:/work:ro" "$IMAGE" \
    /work/belebele_fp8.py compare --dir /out --base "bf16$s" --other "fp8$s" > "$OUT/$name/compare_fp8$s.log" 2>&1 ||
    log "HATA compare $name$s"
}
for spec in "$@"; do
  IFS=: read -r name gpu cpus <<< "$spec"
  log "başla belebele $name (GPU $gpu, CPU $cpus)"
  one "$name" "$gpu" "$cpus" &
done
wait
log "bitti"
