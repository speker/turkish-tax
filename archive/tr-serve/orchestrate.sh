#!/usr/bin/env bash
# Gece sıralayıcısı. Ölçümler yalnız sessiz pencerede: bütün indirmeler bitmeden başlamaz.
# 1) Her model BF16 ve FP8 (gpt-oss zaten MXFP4); 2) iki koşusu %5'ten fazla ayrışan yapılandırmalara
# 3 ek koşu; 3) NCCL tabanı yeniden. Hata veren model atlanır, nedeni log'a yazılır.
set -u
WORK=$HOME/work/tr-serve
TG=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

model_dir() {
  case "${1%-fp8}" in
    turkish-gemma-9b-t1) echo "$TG" ;;
    *) echo "$HOME/models/${1%-fp8}" ;;
  esac
}
extra_args() { [[ "$1" == *-fp8 ]] && echo "--quantization fp8"; }

log "indirmelerin bitmesi bekleniyor"
until grep -q '^\[.*\] bitti' "$WORK/download.log" 2>/dev/null; do sleep 60; done
[ -f "$TG/.tamam" ] || echo bc2e3064994eefab3c9043ff58c5b3c7cbfc6c40 > "$TG/.tamam"

run() {
  local name=$1 dir
  dir=$(model_dir "$name")
  if [ -f "$WORK/results/$name/en_c256_r2.json" ]; then log "$name zaten ölçülmüş"; return; fi
  if [ ! -f "$dir/.tamam" ]; then log "$name: model yok, atlandı"; return; fi
  log "başla $name"
  # shellcheck disable=SC2046
  if bash "$WORK/bench.sh" "$name" "$dir" 1 0 $(extra_args "$name") >"$WORK/bench_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name (ayrıntı: bench_$name.log, results/$name/server.log)"
  fi
}

for name in turkish-gemma-9b-t1 kumru-2b gemma-4-12b-it ministral-3-14b qwen3.8-27b gpt-oss-120b; do
  run "$name"
  [ "$name" = gpt-oss-120b ] || run "$name-fp8"
done

log "tekrar koşuları"
python3 "$WORK/flagged.py" "$WORK/results" | while read -r name lang pair; do
  log "tekrar $name $lang $pair"
  # shellcheck disable=SC2046
  CONC="$pair" LANGS="$lang" REPS="3 4 5" bash "$WORK/bench.sh" "$name" "$(model_dir "$name")" 1 0 \
    $(extra_args "$name") >>"$WORK/bench_$name.log" 2>&1 || log "HATA tekrar $name $lang $pair"
done

log "NCCL tabanı"
OUT_SUFFIX=-iommu-pt bash "$HOME/work/nccl_bench.sh" >"$WORK/nccl.log" 2>&1 || log "HATA NCCL"
log "bitti"
