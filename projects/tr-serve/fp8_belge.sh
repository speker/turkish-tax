#!/usr/bin/env bash
# Belge düzeyinde FP8 kalitesi: servis ölçümünün 562 FLORES+ belgesi, her belge kendi bağlamıyla
# (fp8_quality.py --docs); BF16 iki kez (koşudan koşuya gürültü), FP8 bir kez; bootstrap kaynak belge düzeyinde.
# Hız ölçmez, komşu yükten etkilenmez. Biten etiketler atlanır.
# Kullanım: bash fp8_belge.sh "AD:GPU:CPU" ...   (AD: turkish-gemma-9b-t1 gemma-4-12b-it qwen3.8-27b kumru-2b
#           ministral-3-14b-bf16 ministral-fp8-yerel)
set -u
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
WORK=$HOME/work/tr-serve
M=$HOME/models
IMAGE=tr-serve-vllm:0.33.0-bench
OUT=$WORK/results/fp8-kalite-belge
declare -A PATH_OF=(
  [turkish-gemma-9b-t1]=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf [gemma-4-12b-it]=$M/gemma-4-12b-it
  [qwen3.8-27b]=$M/qwen3.8-27b [kumru-2b]=$M/kumru-2b [ministral-3-14b-bf16]=$M/ministral-3-14b-bf16
)
fp8run() {
  local name=$1 model=$2 label=$3 gpu=$4 cpus=$5
  shift 5
  mkdir -p "$OUT/$name"
  [ -f "$OUT/$name/${label}_en.npz" ] && return 0
  sudo docker run --rm --name "tr-serve-fp8b-$name-$label" --runtime=nvidia --gpus "\"device=$gpu\"" --ipc=host \
    --cpuset-cpus "$cpus" -v "$model:/model:ro" -v "$OUT/$name:/out" -v "$WORK:/work:ro" \
    --entrypoint python3 "$IMAGE" /work/fp8_quality.py run --model /model --label "$label" --text-dir /work \
    --docs /work/data/prompts/docs.json --out /out "$@" > "$OUT/$name/run_$label.log" 2>&1
  [ -f "$OUT/$name/${label}_en.npz" ] && log "tamam $name $label" || log "HATA $name $label"
}
fp8cmp() {
  local name=$1 base=$2 other=$3 cmd
  for cmd in compare paired; do
    sudo docker run --rm --entrypoint python3 -v "$OUT/$name:/out" -v "$WORK:/work:ro" "$IMAGE" \
      /work/fp8_quality.py "$cmd" --dir /out --base "$base" --other "$other" --birim "belge (562)" \
      > "$OUT/$name/${cmd}_$other.log" 2>&1 || log "HATA $cmd $name $other"
  done
}
one() {
  local name=$1 gpu=$2 cpus=$3 extra=()
  if [ "$name" = ministral-fp8-yerel ]; then
    fp8run ministral-3-14b-bf16 "$M/ministral-3-14b" fp8-yerel "$gpu" "$cpus"
    while [ ! -f "$OUT/ministral-3-14b-bf16/bf16_en.npz" ]; do sleep 30; done
    fp8cmp ministral-3-14b-bf16 bf16 fp8-yerel
    return
  fi
  # Gemma 4'ün tokenizer'ı düz metne <bos> eklemiyor (fp8_quality.sh ile aynı)
  [ "$name" = gemma-4-12b-it ] && extra=(--add-bos)
  fp8run "$name" "${PATH_OF[$name]}" bf16 "$gpu" "$cpus" "${extra[@]}"
  fp8run "$name" "${PATH_OF[$name]}" bf16b "$gpu" "$cpus" "${extra[@]}"
  fp8run "$name" "${PATH_OF[$name]}" fp8 "$gpu" "$cpus" --quantization fp8 "${extra[@]}"
  fp8cmp "$name" bf16 bf16b
  fp8cmp "$name" bf16 fp8
}
for spec in "$@"; do
  IFS=: read -r name gpu cpus <<< "$spec"
  log "başla $name (GPU $gpu, CPU $cpus)"
  one "$name" "$gpu" "$cpus" &
done
wait
log "bitti"
