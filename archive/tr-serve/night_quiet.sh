#!/usr/bin/env bash
# Gece sessiz penceresi: başka kiralama kalmayınca quiet kiralama alır ve throughput işlerini
# sırayla yapar: Kumru-2B uzun ölçüm (istem sayısı 4 kat, 3 koşu; 2B model CPU sınırında),
# Ministral 3 14B BF16 reposu (BF16 ve dinamik FP8), NCCL tabanı (iommu=pt ile). Sonda kiralamayı bırakır.
set -u
WORK=$HOME/work/tr-serve
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

# Throughput ölçümleri gece, makine boşken. Başlangıç penceresi 22:00–06:59; ~2,5 sa iş en geç 09:30'da biter.
# lease, GPU'da kiralamasız süreç varken de quiet verir; bu yüzden GPU'ların boş olması ayrıca şart.
night() {
  local h
  h=$(date +%-H)
  [ "$h" -ge 22 ] || [ "$h" -lt 7 ]
}
gpus_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
log "gece penceresi, boş GPU ve quiet kiralama bekleniyor"
id=
while :; do
  if night && gpus_idle; then
    if out=$(~/bin/lease acquire --owner tr-serve --gpus all --mode quiet --hours 4 \
      --note 'Kumru uzun, Ministral BF16, NCCL'); then
      id=$(awk 'NR==1 {print $1}' <<<"$out")
    fi
    if [ -n "$id" ] && gpus_idle; then break; fi
    [ -n "$id" ] && ~/bin/lease release "$id" >/dev/null
    id=
  fi
  sleep 300
done
log "quiet kiralama: $id"

run() {
  local name=$1 dir=$2
  shift 2
  log "başla $name"
  if bash "$WORK/bench.sh" "$name" "$dir" 1 0 "$@" >"$WORK/bench_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
}

export CONC="1:128 16:1024 64:4096 256:16384" REPS="1 2 3"
run kumru-2b-uzun "$HOME/models/kumru-2b"
run kumru-2b-uzun-fp8 "$HOME/models/kumru-2b" --quantization fp8
unset CONC REPS
run ministral-3-14b-bf16 "$HOME/models/ministral-3-14b-bf16"
run ministral-3-14b-bf16-fp8 "$HOME/models/ministral-3-14b-bf16" --quantization fp8

log "NCCL"
OUT_SUFFIX=-iommu-pt bash "$HOME/work/nccl_bench_real.sh" >"$WORK/nccl.log" 2>&1 || log "HATA NCCL"
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
