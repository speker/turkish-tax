#!/usr/bin/env bash
# Ölçekli çıktı: v3 tablo modellerinde Türkçe çıktı uzunluğu 256 × modelin TR/EN token oranı (aynı içerik), İngilizce
# 256. Aynı sunucuda aynı gece TR 256 da ölçülür (v3 tekrarı). 256 eşzamanlılık, iki tekrar, prefix caching kapalı,
# v3 paralel partisinin GPU ve CPU yerleşimi. Sonuçlar results-olcekli/ altına.
# Oranlar results/token_ratio.csv'den: tr_token / en_token × 256, en yakın tamsayı.
# İkinci oturum (ek tekrar, ters sıra, 64 eşzamanlılık):
#   RESULTS_ROOT=~/work/tr-serve/results-olcekli-2 SCALED_CONC="64:1024 256:4096" SCALED_REPS="3 4" ORDER=ters
set -u
WORK=$HOME/work/tr-serve
export RESULTS_ROOT=${RESULTS_ROOT:-$WORK/results-olcekli}
SCALED_CONC=${SCALED_CONC:-"256:4096"}
SCALED_REPS=${SCALED_REPS:-"1 2"}
ORDER=${ORDER:-ileri}
mkdir -p "$RESULTS_ROOT"
TG=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf
M=$HOME/models
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

night() {
  local h
  [ -n "${HEMEN:-}" ] && return 0
  h=$(date +%-H)
  [ "$h" -ge 22 ] || [ "$h" -lt 2 ]
}
gpus_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
log "başlangıç penceresi 22:00-01:59, quiet kiralama ve boş GPU bekleniyor"
id=
while :; do
  until night; do sleep 300; done
  if out=$(~/bin/lease wait --owner tr-serve --gpus all --mode quiet --hours 2 --timeout-min 30 \
    --note "ölçekli çıktı: 4 model, TR 256 x token oranı, $SCALED_CONC, tekrar $SCALED_REPS"); then
    id=$(awk 'NR==1 {print $1}' <<<"$out")
  fi
  if [ -n "$id" ]; then
    for _ in $(seq 1 20); do gpus_idle && break; sleep 30; done
    gpus_idle && break
    log "quiet alındı ama GPU'da süreç kaldı; bırakılıp yeniden denenecek"
    ~/bin/lease release "$id" >/dev/null
    id=
  fi
  sleep 60
done
log "quiet kiralama: $id"

# run AD MODEL_DIZINI GPU CPUSET PORT TR_UZUNLUK
run() {
  local name=$1 dir=$2 gpu=$3 cpus=$4 port=$5 len=$6
  local langs="tr tr:$len en"
  [ "$ORDER" = ters ] && langs="en tr:$len tr"
  log "başla $name (GPU $gpu, CPU $cpus, TR çıktı $len, sıra $langs, $SCALED_CONC, tekrar $SCALED_REPS)"
  if CONC="$SCALED_CONC" REPS="$SCALED_REPS" LANGS="$langs" CPUSET=$cpus PORT=$port \
    bash "$WORK/bench.sh" "$name" "$dir" 1 "$gpu" --no-enable-prefix-caching >"$WORK/bench_olcekli_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
}
run turkish-gemma-9b-t1 "$TG" 1 0-31 8101 357 &
run gemma-4-12b-it "$M/gemma-4-12b-it" 2 32-63 8102 349 &
run ministral-3-14b-bf16 "$M/ministral-3-14b-bf16" 3 64-95 8103 380 &
run qwen3.8-27b "$M/qwen3.8-27b" 4 96-127 8104 365 &
wait
for d in "$RESULTS_ROOT"/*/; do
  grep -q 'enable_prefix_caching=False' "$d/server.log" 2>/dev/null || log "UYARI: $d server.log'da prefix caching kapalı görünmüyor"
done
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
