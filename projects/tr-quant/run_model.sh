#!/usr/bin/env bash
# Tek model için: indir, BF16 GGUF'a dönüştür, TR ve EN imatrix, quant'lar, TR ve EN metinde KLD.
# Kullanım: [GPU_LIST=1,2,3] [CPUSET=96-191] bash run_model.sh REPO_ID REVISION AD
# GPU_LIST: kullanılacak GPU'lar (imatrix ve KLD tabanı ilkinde). CPUSET: CPU işleri bu çekirdeklere sabit.
# Önkoşul: setup.sh ve prepare_text.py (text/ dizini) tamam.
set -euo pipefail
REPO=$1 REV=$2 NAME=$3
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5}
IFS=, read -r -a GPUS <<<"$GPU_LIST"
PIN=()
[ -n "${CPUSET:-}" ] && PIN=(taskset -c "$CPUSET")
WORK=${WORK:-$HOME/work/tr-quant}
BIN=$WORK/llama.cpp/build/bin
PY=$WORK/.venv-convert/bin/python
TEXT=$WORK/text
M=$WORK/models/$NAME
R=$WORK/results/$NAME
CHUNKS_IMATRIX=${CHUNKS_IMATRIX:-200}
CHUNKS_EVAL=${CHUNKS_EVAL:-100}
CTX=512
mkdir -p "$M" "$R"
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

if [ ! -f "$M/bf16.gguf" ]; then
  log "indirme $REPO@$REV"
  "$WORK/.venv/bin/hf" download "$REPO" --revision "$REV" --local-dir "$M/hf" >/dev/null
  # Repoda tokenizer.model yoksa (ör. Gemma 2 türevleri) aynı tabanın SentencePiece modeli verilir:
  # SP_TOKENIZER_FROM=repo@revision. Doğruluğu dönüştürmeden sonra check_vocab.py denetler.
  if [ -n "${SP_TOKENIZER_FROM:-}" ] && [ ! -f "$M/hf/tokenizer.model" ]; then
    log "tokenizer.model: $SP_TOKENIZER_FROM"
    "$WORK/.venv/bin/hf" download "${SP_TOKENIZER_FROM%@*}" tokenizer.model --revision "${SP_TOKENIZER_FROM#*@}" \
      --local-dir "$M/sp" >/dev/null
    cp "$M/sp/tokenizer.model" "$M/hf/tokenizer.model"
  fi
  log "dönüştürme"
  "${PIN[@]}" "$PY" "$WORK/llama.cpp/convert_hf_to_gguf.py" "$M/hf" --outtype bf16 --outfile "$M/bf16.gguf" \
    >"$R/convert.log" 2>&1
  log "vocab denetimi"
  "$PY" "$WORK/check_vocab.py" "$M/bf16.gguf" "$M/hf/tokenizer.json" | tee "$R/check_vocab.log" ||
    { rm -f "$M/bf16.gguf"; log "vocab uyuşmazlığı, durdu"; exit 1; }
fi

for lang in tr en; do
  if [ ! -f "$M/imatrix_$lang.gguf" ]; then
    log "imatrix $lang"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} "${PIN[@]}" "$BIN/llama-imatrix" -m "$M/bf16.gguf" -f "$TEXT/calib_$lang.txt" \
      -o "$M/imatrix_$lang.gguf" -c $CTX --chunks "$CHUNKS_IMATRIX" -ngl 99 >"$R/imatrix_$lang.log" 2>&1
  fi
done

# ad:tip:imatrix
VARIANTS="Q8_0:Q8_0:none Q6_K-tr:Q6_K:tr Q5_K_M-tr:Q5_K_M:tr Q4_K_M-tr:Q4_K_M:tr IQ4_XS-tr:IQ4_XS:tr
Q4_K_M-en:Q4_K_M:en IQ4_XS-en:IQ4_XS:en Q4_K_M-none:Q4_K_M:none IQ3_M-tr:IQ3_M:tr IQ3_M-en:IQ3_M:en"
for v in $VARIANTS; do
  IFS=: read -r vname qtype imx <<<"$v"
  out="$M/$vname.gguf"
  [ -f "$out" ] && continue
  log "quantize $vname"
  args=()
  [ "$imx" != none ] && args=(--imatrix "$M/imatrix_$imx.gguf")
  "${PIN[@]}" "$BIN/llama-quantize" "${args[@]}" "$M/bf16.gguf" "$out" "$qtype" >"$R/quantize_$vname.log" 2>&1
done

for lang in tr en; do
  base="$M/kld_base_$lang.bin"
  if [ ! -f "$base" ]; then
    log "KLD tabanı $lang"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} "${PIN[@]}" "$BIN/llama-perplexity" -m "$M/bf16.gguf" -f "$TEXT/eval_$lang.txt" \
      -c $CTX --chunks "$CHUNKS_EVAL" -ngl 99 --kl-divergence-base "$base" >"$R/ppl_bf16_$lang.log" 2>&1
  fi
done

jobs_list=()
for v in $VARIANTS; do
  vname=${v%%:*}
  for lang in tr en; do jobs_list+=("$vname:$lang"); done
done
slot=0
for job in "${jobs_list[@]}"; do
  vname=${job%%:*} lang=${job##*:}
  logf="$R/kld_${vname}_$lang.log"
  [ -s "$logf" ] && grep -q 'Mean    KLD' "$logf" && continue
  gpu=${GPUS[$slot]}
  log "KLD $vname $lang (GPU $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu "${PIN[@]}" "$BIN/llama-perplexity" -m "$M/$vname.gguf" -c $CTX --chunks "$CHUNKS_EVAL" \
    -ngl 99 --kl-divergence-base "$M/kld_base_$lang.bin" --kl-divergence >"$logf" 2>&1 &
  slot=$(( (slot + 1) % ${#GPUS[@]} ))
  [ "$slot" -eq 0 ] && wait
done
wait
for f in "$M"/*.gguf; do printf '%s\t%s\n' "$(basename "$f")" "$(stat -c %s "$f")"; done >"$R/sizes.tsv"
log "bitti: $R"
