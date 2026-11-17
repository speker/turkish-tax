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
# IGNORE_EOS=0: doğal bitiş (çıktı uzunluğu üst sınır olur); RATE=<istek/sn>: açık döngü Poisson gelişleri;
# DETAILED=1: istek başına çıktı uzunlukları sonuca yazılır.
IGNORE_EOS=${IGNORE_EOS:-1}
RATE=${RATE:-inf}
DETAILED=${DETAILED:-0}
# ORDER=alt: çift numaralı tekrarlarda dil sırası ters çevrilir (sıra etkisi için).
# BENCH_ARGS: istemciye ek argüman, ör. "--temperature 0". vLLM 0.33 istemcisi sıcaklık göndermez; boşsa her model
# kendi generation_config varsayılanıyla örneklenir.
ORDER=${ORDER:-fixed}
BENCH_ARGS=${BENCH_ARGS:-}
# DOCKER_ARGS: docker run'a ek argüman, ör. "-e NCCL_NVLS_ENABLE=0" (NVSwitch bölümü NVLS desteklemeyen makinelerde).
DOCKER_ARGS=${DOCKER_ARGS:-}
# GPU'ların CPU yakınlığı (nvidia-smi topo -m): 0-47
CPUSET=${CPUSET:-0-47}
# CLIENT_CPUSET: verilirse istemci sunucu konteynerinde değil, bu CPU'lara sabit ayrı bir konteynerde çalışır
# (istemci ile sunucunun aynı çekirdekleri paylaşmasının etkisini ölçmek için).
CLIENT_CPUSET=${CLIENT_CPUSET:-}
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
  --ulimit nofile=65536:65536 $DOCKER_ARGS \
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
  echo " \"ignore_eos\": $IGNORE_EOS, \"request_rate\": \"$RATE\", \"order\": \"$ORDER\", \"bench_args\": \"$BENCH_ARGS\","
  echo " \"docker_args\": \"$DOCKER_ARGS\", \"client_cpuset\": \"$CLIENT_CPUSET\","
  echo " \"driver\": \"$(cat /sys/module/nvidia/version)\", \"output_len\": $OUT_LEN, \"date\": \"$(date -Is)\"}"
} > "$RES/ortam.json"

bench() {
  local len=$OUT_LEN
  [[ $1 == *:* ]] && len=${1##*:}
  local flags=()
  [ "$IGNORE_EOS" = 1 ] && flags+=(--ignore-eos)
  [ "$DETAILED" = 1 ] && flags+=(--save-detailed)
  # shellcheck disable=SC2206
  flags+=($BENCH_ARGS)
  local client=(sudo docker exec "$cname" vllm) url=http://127.0.0.1:8000
  if [ -n "$CLIENT_CPUSET" ]; then
    client=(sudo docker run --rm --network host --cpuset-cpus "$CLIENT_CPUSET" --ulimit nofile=65536:65536
      -v "$MODEL:/model:ro" -v "$WORK/data/prompts:/data:ro" -v "$RES:/results" --entrypoint vllm "$IMAGE")
    url=http://127.0.0.1:$PORT
  fi
  "${client[@]}" bench serve --backend vllm --base-url "$url" \
    --model "$NAME" --tokenizer /model --dataset-name custom --dataset-path "/data/prompts_${1%%:*}.jsonl" \
    --custom-output-len "$len" "${flags[@]}" --request-rate "$RATE" --disable-shuffle --num-prompts "$3" \
    --max-concurrency "$2" --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
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
  langs=$LANGS
  [ "$ORDER" = alt ] && [ $((rep % 2)) -eq 0 ] && langs=$(printf '%s\n' $LANGS | tac | xargs)
  for pair in $CONC; do
    c=${pair%%:*} n=${pair##*:}
    for lang in $langs; do
      log "koşu $rep: $lang c=$c n=$n"
      bench "$lang" "$c" "$n" "${lang/:/-o}_c${c}_r${rep}.json"
    done
  done
done
log "bitti: $RES"
