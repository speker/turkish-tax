#!/usr/bin/env bash
# Servis ölçümündeki modelleri sabit revision'larından ~/models/<ad> altına sırayla indirir (küçükten büyüğe).
# Turkish-Gemma GGUF hattıyla paylaşılır (~/work/tr-quant/models/turkish-gemma-9b-t1/hf), burada indirilmez.
set -u
export PATH=$HOME/.local/bin:$PATH
HF=$HOME/work/tr-quant/.venv/bin/hf
DEST=$HOME/models
mkdir -p "$DEST"
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
log "indirmeler başlıyor"

MODELS="
kumru-2b vngrs/Kumru-2B 0b3cf37f657ef4013bdc99598545f5d317bd7397
gemma-4-12b-it google/gemma-4-12B-it 43c9d014b231474fa7212dbb48a762566c2b255d
ministral-3-14b mistralai/Ministral-3-14B-Instruct-2512 29439f81c2be264d8d393273f99e7db9c0961120
ministral-3-14b-bf16 mistralai/Ministral-3-14B-Instruct-2512-BF16 3cea74c1ebaf5ce5f5a2553de470e2ceab825142
qwen3.8-27b Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
gpt-oss-120b openai/gpt-oss-120b b5c939de8f754692c1647ca79fbf85e8c1e70f8a
"
echo "$MODELS" | while read -r name repo rev; do
  [ -z "$name" ] && continue
  if [ -f "$DEST/$name/.tamam" ]; then log "$name zaten var"; continue; fi
  log "indirme $repo@$rev"
  for attempt in 1 2 3; do
    if "$HF" download "$repo" --revision "$rev" --local-dir "$DEST/$name" \
        --exclude "original/*" --exclude "metal/*" --exclude "*.gguf" >"$DEST/$name.log" 2>&1; then
      weights=$(find "$DEST/$name" -maxdepth 1 -name '*.safetensors' | wc -l)
      bytes=$(du -sb "$DEST/$name" | cut -f1)
      if [ "$weights" -gt 0 ] && [ "$bytes" -gt 1000000000 ]; then
        echo "$rev" >"$DEST/$name/.tamam"
        log "$name tamam ($(du -sh "$DEST/$name" | cut -f1), $weights safetensors)"
        break
      fi
      log "$name indirildi görünüyor ama ağırlık yok ($weights dosya, $bytes bayt)"
    fi
    log "$name deneme $attempt başarısız"
    sleep 30
  done
done
missing=0
for name in $(echo "$MODELS" | awk 'NF {print $1}'); do
  [ -f "$DEST/$name/.tamam" ] || { log "$name eksik"; missing=1; }
done
log "bitti"
exit $missing
