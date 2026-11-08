#!/usr/bin/env bash
# FP8'in Türkçe kalitesi (fp8_quality.py): shared kiralamayla GPU 2-5, her GPU'da bir model kuyruğu.
# Kalite ölçümü; hız ölçmediği için quiet gerekmez. Kullanım: bash fp8_quality.sh
set -u
WORK=$HOME/work/tr-serve
IMAGE=tr-serve-vllm:0.33.0-bench
TEXT=$HOME/work/tr-quant/text
OUT=$WORK/results/fp8-kalite
TG=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

LEASE_GPUS=${LEASE_GPUS:-2,3,4,5}
id=$(~/bin/lease wait --owner tr-serve --gpus "$LEASE_GPUS" --mode shared --hours 3 \
  --note "FP8 Türkçe kalite (prompt_logprobs, FLORES+) ${ONLY:-}" --timeout-min 240 | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "kiralama alınamadı"; exit 1; }
log "shared kiralama: $id (GPU $LEASE_GPUS)"

# run AD MODEL_DIZINI ETİKET GPU CPUSET [--quantization fp8]
run() {
  local name=$1 model=$2 label=$3 gpu=$4 cpus=$5
  shift 5
  mkdir -p "$OUT/$name"
  [ -f "$OUT/$name/${label}_en.npz" ] && return 0
  log "başla $name $label (GPU $gpu)"
  if sudo docker run --rm --name "tr-serve-fp8q-$name-$label" --runtime=nvidia --gpus "\"device=$gpu\"" --ipc=host \
    --cpuset-cpus "$cpus" -v "$model:/model:ro" -v "$TEXT:/text:ro" -v "$OUT/$name:/out" -v "$WORK:/work:ro" \
    --entrypoint python3 "$IMAGE" /work/fp8_quality.py run --model /model --label "$label" --text-dir /text \
    --out /out "$@" >"$OUT/$name/run_$label.log" 2>&1 || [ -f "$OUT/$name/${label}_en.npz" ]; then
    log "tamam $name $label"
  else
    log "HATA $name $label (son satırlar):"
    tail -n 3 "$OUT/$name/run_$label.log" | sed 's/^/    /'
  fi
}
compare() {
  local name=$1 base=$2 other=$3
  sudo docker run --rm --entrypoint python3 -v "$OUT/$name:/out" -v "$WORK:/work:ro" "$IMAGE" \
    /work/fp8_quality.py compare --dir /out --base "$base" --other "$other" >"$OUT/$name/compare_$other.log" 2>&1 ||
    log "HATA compare $name $other"
}
pair() {
  local name=$1 model=$2 gpu=$3 cpus=$4
  shift 4
  run "$name" "$model" bf16 "$gpu" "$cpus" "$@"
  run "$name" "$model" fp8 "$gpu" "$cpus" --quantization fp8 "$@"
  compare "$name" bf16 fp8
}

# ONLY=gemma-4-12b-it: yalnız o modeli koşar
if [ -n "${ONLY:-}" ]; then
  case $ONLY in
    gemma-4-12b-it) pair gemma-4-12b-it "$HOME/models/gemma-4-12b-it" 3 120-143 --add-bos ;;
    *) log "bilinmeyen ONLY=$ONLY" ;;
  esac
  ~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
  exit 0
fi

pair turkish-gemma-9b-t1 "$TG" 2 96-119 &
# Gemma 4'ün tokenizer'ı düz metne <bos> eklemiyor (add_bos_token tanımsız)
pair gemma-4-12b-it "$HOME/models/gemma-4-12b-it" 3 120-143 --add-bos &
pair qwen3.8-27b "$HOME/models/qwen3.8-27b" 4 144-167 &
{
  pair kumru-2b "$HOME/models/kumru-2b" 5 168-191
  pair ministral-3-14b-bf16 "$HOME/models/ministral-3-14b-bf16" 5 168-191
  # Ana repo üreticinin statik FP8 ağırlıkları; taban yine BF16 reposu (tokenizer aynıysa compare geçer)
  run ministral-3-14b-bf16 "$HOME/models/ministral-3-14b" fp8-yerel 5 168-191
  compare ministral-3-14b-bf16 bf16 fp8-yerel
} &
wait
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
