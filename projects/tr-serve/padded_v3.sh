#!/usr/bin/env bash
# Gemma'da kalan yük çarpanının sınaması (Post yorumu, 9 Kasım): aynı sunucuda dört koşu, 256 eşzamanlılık, iki tekrar.
#   tr:N        Türkçe, çıktı N = 256 × token oranı (night_scaled_v3.sh ile aynı)
#   enpad-X:N   İngilizce istem Türkçe istemin token sayısına doldurulmuş (make_padded_prompts.py), çıktı N
#   en:N        İngilizce istem, çıktı N
#   en          İngilizce istem, çıktı 256
# Bağlam uzunluğu açıklıyorsa enpad ile tr aynı hızda olur. v3 paralel partisinin GPU ve CPU yerleşimi.
# Başlangıç: WAIT_JOB verilirse o done-claims işi bitip GPU'lar boşalınca; sonra quiet kiralama.
set -u
WORK=$HOME/work/tr-serve
export RESULTS_ROOT=$WORK/results-dolgu
mkdir -p "$RESULTS_ROOT"
TG=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf
M=$HOME/models
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
gpus_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }

if [ -n "${WAIT_JOB:-}" ]; then
  log "$WAIT_JOB bitişi bekleniyor"
  until [ "$(bash "$HOME/work/done-claims/job_done.sh" "$WAIT_JOB" | awk '{print $2}')" -ge 1 ]; do sleep 120; done
  log "$WAIT_JOB bitti"
fi
id=$(~/bin/lease wait --owner tr-serve --gpus all --mode quiet --hours 2 --timeout-min 60 \
  --note 'dolgulu istem sınaması: 4 model, 4 koşu, 256 eşzamanlılık (~50 dk)' | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "quiet kiralama alınamadı"; exit 1; }
for _ in $(seq 1 20); do gpus_idle && break; sleep 30; done
gpus_idle || { log "GPU'da süreç kaldı; bırakıldı"; ~/bin/lease release "$id"; exit 1; }
log "quiet kiralama: $id"

# run AD MODEL_DIZINI GPU CPUSET PORT N DOLGU_ADI
run() {
  local name=$1 dir=$2 gpu=$3 cpus=$4 port=$5 n=$6 pad=$7
  log "başla $name (GPU $gpu, CPU $cpus, N $n)"
  if CONC="256:4096" REPS="1 2" LANGS="tr:$n enpad-$pad:$n en:$n en" CPUSET=$cpus PORT=$port \
    bash "$WORK/bench.sh" "$name" "$dir" 1 "$gpu" --no-enable-prefix-caching >"$WORK/bench_dolgu_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
}
run turkish-gemma-9b-t1 "$TG" 1 0-31 8101 357 tg &
run gemma-4-12b-it "$M/gemma-4-12b-it" 2 32-63 8102 349 g4 &
run ministral-3-14b-bf16 "$M/ministral-3-14b-bf16" 3 64-95 8103 380 mi &
run qwen3.8-27b "$M/qwen3.8-27b" 4 96-127 8104 365 qw &
wait
for d in "$RESULTS_ROOT"/*/; do
  grep -q 'enable_prefix_caching=False' "$d/server.log" 2>/dev/null || log "UYARI: $d prefix caching kapalı görünmüyor"
done
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
