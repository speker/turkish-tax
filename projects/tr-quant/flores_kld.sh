#!/usr/bin/env bash
# Aynı içerikte TR ve EN KLD: FLORES+ dev ve devtest (2.009 paralel cümle), run_model.sh'in ürettiği GGUF'larla.
# Vikipedi değerlendirmesinde TR ve EN metinler farklı makaleler; burada iki dilde içerik birebir aynı.
# Kullanım: [GPU_LIST=2,3,4,5] [CPUSET=96-191] bash flores_kld.sh AD [AD ...]
# Önkoşul: run_model.sh o model için bitmiş; FLORES+ jsonl dosyaları $FLORES/{dev,devtest}/ altında.
set -euo pipefail
GPU_LIST=${GPU_LIST:-2,3,4,5}
IFS=, read -r -a GPUS <<<"$GPU_LIST"
PIN=()
[ -n "${CPUSET:-}" ] && PIN=(taskset -c "$CPUSET")
WORK=${WORK:-$HOME/work/tr-quant}
BIN=$WORK/llama.cpp/build/bin
FLORES=${FLORES:-$HOME/work/tr-serve/data/flores}
TEXT=$WORK/text
CTX=512
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

for lang in tr en; do
  out=$TEXT/flores_$lang.txt
  [ -s "$out" ] && continue
  code=eng_Latn
  [ "$lang" = tr ] && code=tur_Latn
  python3 - "$FLORES" "$code" "$out" <<'EOF'
import json
import sys

root, code, out = sys.argv[1:]
lines = []
for split in ("dev", "devtest"):
    with open(f"{root}/{split}/{code}.jsonl", encoding="utf-8") as fh:
        rows = sorted((json.loads(line) for line in fh), key=lambda r: r["id"])
    lines += [r["text"].strip() for r in rows]
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{code}: {len(lines)} cümle -> {out}")
EOF
done
[ "$(wc -l <"$TEXT/flores_tr.txt")" -eq "$(wc -l <"$TEXT/flores_en.txt")" ] || { log "TR ve EN satır sayısı farklı"; exit 1; }

for NAME in "$@"; do
  M=$WORK/models/$NAME
  R=$WORK/results/$NAME/flores
  mkdir -p "$R"
  mapfile -t variants < <(cd "$M" && ls ./*.gguf | sed 's|^\./||; s|\.gguf$||' | grep -v -e '^bf16$' -e '^imatrix_')
  for lang in tr en; do
    base=$M/kld_base_flores_$lang.bin
    [ -f "$base" ] && continue
    log "$NAME KLD tabanı $lang"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} "${PIN[@]}" "$BIN/llama-perplexity" -m "$M/bf16.gguf" -f "$TEXT/flores_$lang.txt" \
      -c $CTX -ngl 99 --kl-divergence-base "$base" >"$R/ppl_bf16_$lang.log" 2>&1
  done
  slot=0
  for vname in "${variants[@]}"; do
    for lang in tr en; do
      logf="$R/kld_${vname}_$lang.log"
      [ -s "$logf" ] && grep -q 'Mean    KLD' "$logf" && continue
      gpu=${GPUS[$slot]}
      log "$NAME KLD $vname $lang (GPU $gpu)"
      CUDA_VISIBLE_DEVICES=$gpu "${PIN[@]}" "$BIN/llama-perplexity" -m "$M/$vname.gguf" -c $CTX -ngl 99 \
        --kl-divergence-base "$M/kld_base_flores_$lang.bin" --kl-divergence >"$logf" 2>&1 &
      slot=$(((slot + 1) % ${#GPUS[@]}))
      [ "$slot" -eq 0 ] && wait
    done
  done
  wait
  cp "$WORK/results/$NAME/sizes.tsv" "$R/sizes.tsv"
  log "bitti: $R"
done
