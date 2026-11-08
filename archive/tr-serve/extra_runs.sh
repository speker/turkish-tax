#!/usr/bin/env bash
# Ek ölçüm: Ministral 3 14B'nin ana reposu FP8 (statik) ağırlıkla geliyor; BF16 karşılaştırması için ayrı
# BF16 reposu indirilir (CPU 96-191'e sabit, çalışan ölçümü etkilemesin) ve kumru_long bitince ölçülür.
set -u
WORK=$HOME/work/tr-serve
HF=$HOME/work/tr-quant/.venv/bin/hf
NAME=ministral-3-14b-bf16
DIR=$HOME/models/$NAME
REPO=mistralai/Ministral-3-14B-Instruct-2512-BF16
REV=3cea74c1ebaf5ce5f5a2553de470e2ceab825142
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

if [ ! -f "$DIR/.tamam" ]; then
  log "indirme $REPO@$REV"
  taskset -c 96-191 "$HF" download "$REPO" --revision "$REV" --local-dir "$DIR" \
    --exclude "original/*" --exclude "*.gguf" >"$HOME/models/$NAME.log" 2>&1 || { log "indirme HATA"; exit 1; }
  [ "$(find "$DIR" -maxdepth 1 -name '*.safetensors' | wc -l)" -gt 0 ] || { log "ağırlık yok"; exit 1; }
  echo "$REV" >"$DIR/.tamam"
  log "indirildi ($(du -sh "$DIR" | cut -f1))"
fi
python3 "$WORK/model_dtype.py" "$DIR"

until grep -q '\] bitti' "$WORK/kumru_long.log" 2>/dev/null; do sleep 60; done
for name in "$NAME" "$NAME-fp8"; do
  extra=()
  [[ "$name" == *-fp8 ]] && extra=(--quantization fp8)
  log "başla $name"
  if bash "$WORK/bench.sh" "$name" "$DIR" 1 0 "${extra[@]}" >"$WORK/bench_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
done
log "bitti"
