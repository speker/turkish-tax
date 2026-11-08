#!/usr/bin/env bash
# Tek model için vLLM servis ölçümü: TR ve EN paralel istemler, eşzamanlılık 1/16/64/256, her biri iki koşu.
# Kullanım: bash bench.sh AD MODEL_DIZINI TP GPU_LISTESI [ek vLLM argümanları...]
#   örnek: bash bench.sh turkish-gemma-9b-t1 ~/models/x 1 0
#          bash bench.sh qwen3.8-27b-fp8 ~/models/qwen3.8-27b 2 0,1 --quantization fp8
set -euo pipefail
NAME=$1 MODEL=$2 TP=$3 GPUS=$4
shift 4
EXTRA=("$@")
# Dockerfile.bench: resmi vLLM 0.33.0 imajı (digest aşağıda) + pandas 2.3.3
BASE_IMAGE=vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
IMAGE=tr-serve-vllm:0.33.0-bench
WORK=$HOME/work/tr-serve
RES=${RESULTS_ROOT:-$WORK/results}/$NAME
PORT=${PORT:-8100}
OUT_LEN=256
# Tekrar koşuları için dışarıdan daraltılabilir: CONC="64:1024" LANGS="tr" REPS="3 4 5"
# LANGS öğesi "dil:uzunluk" ise o koşu farklı çıktı uzunluğuyla yapılır ve dosya adı "dil-o<uzunluk>" olur
# (ör. LANGS="tr tr:357 en" -> tr_c256_r1, tr-o357_c256_r1, en_c256_r1).
# İstem sayısı eşzamanlılığın en az 16 katı: kısa koşularda başlama/bitiş dalgası sonucu belirliyor.
# 562 belgeden fazlası vLLM'in sabit tohumlu oversampling'iyle tamamlanır; TR ve EN aynı belgeleri alır.
CONC=${CONC:-"1:32 16:256 64:1024 256:4096"}
LANGS=${LANGS:-"tr en"}
REPS=${REPS:-"1 2"}
# GPU'ların CPU yakınlığı (nvidia-smi topo -m): 0-47
CPUSET=${CPUSET:-0-47}
mkdir -p "$RES"
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

busy=$(nvidia-smi -i "$GPUS" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -eq 0 ] || { log "GPU $GPUS'da $busy süreç var; ölçüm başlamadı"; exit 1; }
# Başka GPU'larda o an çalışan işler sonuca yazılır (paralel ölçümde karşılaştırma için).
others=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits |
  awk -F', ' -v g=",$GPUS," 'index(g, ","$1",") == 0 && $2 > 5 {printf "%s%s", s, $1; s=","}')

cname=tr-serve-$NAME
sudo docker rm -f "$cname" >/dev/null 2>&1 || true
# --ulimit: 1024 eşzamanlılıkta benchmark istemcisi varsayılan dosya sınırına takılıyordu (Errno 24)
sudo docker run -d --name "$cname" --runtime=nvidia --gpus "\"device=$GPUS\"" --ipc=host --cpuset-cpus "$CPUSET" \
  --ulimit nofile=65536:65536 \
  -p "127.0.0.1:$PORT:8000" -v "$MODEL:/model:ro" -v "$WORK/data/prompts:/data:ro" -v "$RES:/results" \
  "$IMAGE" --model /model --served-model-name "$NAME" --tensor-parallel-size "$TP" \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --seed 0 "${EXTRA[@]}" >/dev/null
trap 'sudo docker logs "$cname" > "$RES/server.log" 2>&1; sudo docker rm -f "$cname" >/dev/null 2>&1' EXIT

log "sunucu bekleniyor ($NAME, TP=$TP, GPU=$GPUS ${EXTRA[*]})"
for i in $(seq 1 120); do
  curl -sf "127.0.0.1:$PORT/health" >/dev/null && break
  sudo docker inspect -f '{{.State.Running}}' "$cname" | grep -q true || { log "sunucu durdu"; exit 1; }
  sleep 10
done
curl -sf "127.0.0.1:$PORT/health" >/dev/null || { log "sunucu 20 dakikada hazır olmadı"; exit 1; }

{ [ ! -f "$RES/ortam.json" ] || [[ " $REPS " == *" 1 "* ]]; } && {
  echo "{\"name\": \"$NAME\", \"base_image\": \"$BASE_IMAGE\","
  echo " \"image_id\": \"$(sudo docker image inspect --format '{{.Id}}' "$IMAGE")\", \"tp\": $TP, \"gpus\": \"$GPUS\","
  echo " \"cpuset\": \"$CPUSET\", \"kernel_cmdline\": \"$(cat /proc/cmdline)\", \"conc\": \"$CONC\","
  echo " \"diger_mesgul_gpular\": \"$others\","
  echo " \"extra_args\": \"${EXTRA[*]}\", \"model_revision\": \"$(cat "$MODEL/.tamam" 2>/dev/null || echo '-')\","
  echo " \"driver\": \"$(cat /sys/module/nvidia/version)\", \"output_len\": $OUT_LEN, \"date\": \"$(date -Is)\"}"
} > "$RES/ortam.json"

bench() {
  local len=$OUT_LEN
  [[ $1 == *:* ]] && len=${1##*:}
  sudo docker exec "$cname" vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
    --model "$NAME" --tokenizer /model --dataset-name custom --dataset-path "/data/prompts_${1%%:*}.jsonl" \
    --custom-output-len "$len" --ignore-eos --disable-shuffle --num-prompts "$3" --max-concurrency "$2" \
    --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir /results --result-filename "$4" >"$RES/${4%.json}.log" 2>&1
}

log "ısınma"
bench tr 16 32 warmup.json
# WARMUP_FULL=1: bütün eşzamanlılık ve dillerde ölçülmeyen bir tur (gpt-oss-120b'de sunucu açılışından sonraki ilk
# tur 16+ eşzamanlılıkta %18-40 yavaş). Dosyalar warmup_* adıyla yazılır, özete girmez.
if [ -n "${WARMUP_FULL:-}" ]; then
  for pair in $CONC; do
    for lang in $LANGS; do bench "$lang" "${pair%%:*}" "${pair##*:}" "warmup_${lang}_c${pair%%:*}.json"; done
  done
fi
for rep in $REPS; do
  for pair in $CONC; do
    c=${pair%%:*} n=${pair##*:}
    for lang in $LANGS; do
      log "koşu $rep: $lang c=$c n=$n"
      bench "$lang" "$c" "$n" "${lang/:/-o}_c${c}_r${rep}.json"
    done
  done
done
log "bitti: $RES"
