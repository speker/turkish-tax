#!/usr/bin/env bash
# padded_backend.sh'teki beş hücre, bu kez her sunucu makinede tek başına (Post yorumu, 10 Kasım 16:34: beş sunucu
# aynı makineyi paylaşırken görülen Triton farkı, her biri tek başına koşunca da sürüyor mu?). Hücreler sırayla aynı
# GPU'da ve aynı CPU kümesinde koşar; istem, çıktı uzunlukları (256 ve N), 256 eşzamanlılık ve beş tekrar
# padded_backend.sh ile aynı. Başlamadan önce yük ortalaması 8'in altına inene kadar beklenir (en çok 30 dk); koşu
# boyunca yük ortalaması ve GPU kullanımı 30 saniyede bir host_load.log'a yazılır.
set -u
WORK=$HOME/work/tr-serve
export RESULTS_ROOT=$WORK/results-cekirdek-tek
mkdir -p "$RESULTS_ROOT"
TG=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf
G4=$HOME/models/gemma-4-12b-it
GPU=${GPU:-0}
CPUS=${CPUS:-0-31}
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
gpus_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }

for _ in $(seq 1 60); do
  awk -v l="$(cut -d' ' -f1 /proc/loadavg)" 'BEGIN {exit !(l < 8)}' && break
  sleep 30
done
log "yük ortalaması: $(cut -d' ' -f1-3 /proc/loadavg)"

id=$(~/bin/lease wait --owner tr-serve --gpus all --mode quiet --hours 3 --timeout-min "${WAIT_MIN:-60}" \
  --note 'çekirdek ayrımı, her sunucu tek başına: 2 model, 5 hücre sırayla, 256 eşzamanlılık, 5 tekrar (~2,5 sa)' |
  awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "quiet kiralama alınamadı"; exit 1; }
for _ in $(seq 1 20); do gpus_idle && break; sleep 30; done
gpus_idle || { log "GPU'da süreç kaldı; bırakıldı"; ~/bin/lease release "$id"; exit 1; }
log "quiet kiralama: $id"

(
  while :; do
    printf '%s %s %s\n' "$(date +%T)" "$(cut -d' ' -f1-3 /proc/loadavg)" \
      "$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | tr '\n' ';')"
    sleep 30
  done
) >"$RESULTS_ROOT/host_load.log" 2>&1 &
watcher=$!

# run AD MODEL_DIZINI N [ek vLLM argümanları]
run() {
  local name=$1 dir=$2 n=$3
  shift 3
  log "başla $name (GPU $GPU, CPU $CPUS, N $n, $*)"
  if CONC="256:4096" REPS="1 2 3 4 5" LANGS="en:$n en" CPUSET=$CPUS PORT=8111 \
    bash "$WORK/bench.sh" "$name" "$dir" 1 "$GPU" --no-enable-prefix-caching "$@" >"$WORK/bench_tek_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
}
run turkish-gemma-9b-t1 "$TG" 357
run gemma-4-12b-it "$G4" 349
run turkish-gemma-9b-t1-fa4 "$TG" 357 --attention-config.flash_attn_version=4
run turkish-gemma-9b-t1-triton "$TG" 357 --attention-backend TRITON_ATTN
run gemma-4-12b-it-triton "$G4" 349 --attention-backend TRITON_ATTN
kill "$watcher"
for d in "$RESULTS_ROOT"/*/; do
  grep -h -E 'attention backend out of|FlashAttention version' "$d/server.log" 2>/dev/null |
    sed -E "s|^.*\] |$(basename "$d"): |"
done >"$RESULTS_ROOT/backends.txt"
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
