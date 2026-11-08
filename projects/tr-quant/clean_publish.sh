#!/usr/bin/env bash
# Hub'a yüklenecek GGUF dosyalarını clean_gguf_meta.py ile publish/<model>/ altına yazar.
# Her dosyanın sonucu publish/<model>.clean_meta.jsonl'de; publish.py upload-remote bu kaydı manifest olarak
# kullanır. Klasörde yalnız .gguf durur.
set -euo pipefail
WORK=${WORK:-$HOME/work/tr-quant}
PY=$WORK/.venv-convert/bin/python
# llama.cpp b12116 ağacındaki gguf-py (pip'teki 0.19.0'da GGUFWriter.remove_key yok)
export PYTHONPATH=$WORK/llama.cpp/gguf-py
FILES="Q8_0 Q6_K-tr Q5_K_M-tr Q4_K_M-tr IQ4_XS-tr IQ3_M-tr imatrix_tr"
id=$(~/bin/lease acquire --owner tr-quant --gpus none --mode shared --hours 2 \
  --note 'GGUF metadata temizliği, disk G/Ç' | awk 'NR==1 {print $1}')
[ -n "$id" ] || { echo "kiralama alınamadı" >&2; exit 1; }
trap '~/bin/lease release "$id"' EXIT

clean() {
  local model=$1 name=$2 out=$WORK/publish/$1 log=$WORK/publish/$1.clean_meta.jsonl
  rm -rf "$out"
  mkdir -p "$out"
  : >"$log"
  for f in $FILES; do
    "$PY" "$WORK/clean_gguf_meta.py" "$WORK/models/$model/$f.gguf" "$out/$f.gguf" --name "$name" >>"$log"
  done
  echo "$model: $(wc -l <"$log") dosya temiz"
}

clean turkish-gemma-9b-t1 Turkish-Gemma-9b-T1
clean trendyol-llm-8b-t1 Trendyol-LLM-8B-T1
