#!/usr/bin/env bash
# Ölçülen ve temizlenmiş Q4_K_M dosyası llama.cpp'de aynı sonucu veriyor mu: CPU'da kısa llama-perplexity.
# Çıktı publish/<model>.clean_load_test.txt (komut, sürüm, iki dosyanın SHA-256'sı, PPL).
set -euo pipefail
WORK=${WORK:-$HOME/work/tr-quant}
cd "$WORK"
id=$(~/bin/lease acquire --owner tr-quant --gpus none --mode shared --hours 1 \
  --note 'temizlenmiş GGUF yükleme testi (CPU)' | awk 'NR==1 {print $1}')
[ -n "$id" ] || { echo "kiralama alınamadı" >&2; exit 1; }
trap '~/bin/lease release "$id" >/dev/null' EXIT
BIN=llama.cpp/build/bin/llama-perplexity
ARGS=(-f text/flores_tr.txt -c 512 --chunks 3 -ngl 0 -t 64)

for m in turkish-gemma-9b-t1 trendyol-llm-8b-t1; do
  {
    echo "llama.cpp: $("$BIN" --version 2>&1 | grep -m1 version)"
    echo "komut: CUDA_VISIBLE_DEVICES= $BIN -m <dosya> ${ARGS[*]}"
    for f in models/$m/Q4_K_M-tr.gguf publish/$m/Q4_K_M-tr.gguf; do
      ppl=$(CUDA_VISIBLE_DEVICES= taskset -c 96-159 "$BIN" -m "$f" "${ARGS[@]}" 2>&1 |
        grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*')
      echo "$f sha256 $(sha256sum "$f" | cut -d' ' -f1) | $ppl"
    done
  } >"publish/$m.clean_load_test.txt"
  cat "publish/$m.clean_load_test.txt"
done
