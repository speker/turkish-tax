#!/usr/bin/env bash
# GGUF kuyruğu: GPU 1-5 ve CPU 96-191 (GPU 0 ve CPU 0-47 o sırada servis ölçümünde).
# Kullanım: [QUEUE=dosya] bash queue_gguf.sh   (dosya satırı: REPO REVISION AD)
set -u
WORK=$HOME/work/tr-quant
export GPU_LIST=${GPU_LIST:-1,2,3,4,5} CPUSET=${CPUSET:-96-191}
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
default_queue() {
  cat <<'EOF'
Trendyol/Trendyol-LLM-8B-T1 1aeda7229465d71662aacd26e6c6e8f167f683a7 trendyol-llm-8b-t1
EOF
}
# Kumru-2B bu kuyrukta yok: llama.cpp'de pre-tokenizer'ı tanımlı değil, kumru_pipeline.sh yamayla çalıştırır.
# Turkish-Gemma-9b-T1 (repoda tokenizer.model yok, Gemma 2'ninki verilir):
#   SP_TOKENIZER_FROM=unsloth/gemma-2-9b@9145841c83add11c48155cd7bb87d8656703f997 \
#   bash run_model.sh ytu-ce-cosmos/Turkish-Gemma-9b-T1 bc2e3064994eefab3c9043ff58c5b3c7cbfc6c40 turkish-gemma-9b-t1
rm -f "$WORK/queue_failed.txt"
{ if [ -n "${QUEUE:-}" ]; then cat "$QUEUE"; else default_queue; fi; } | while read -r repo rev name; do
  [ -z "$repo" ] && continue
  log "başla $name"
  if bash "$WORK/run_model.sh" "$repo" "$rev" "$name" >"$WORK/run_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name; son satırlar:"
    tail -n 5 "$WORK/run_$name.log" | sed 's/^/    /'
    echo "$name" >>"$WORK/queue_failed.txt"
  fi
done
log "bitti"
[ ! -s "$WORK/queue_failed.txt" ]
