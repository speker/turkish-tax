#!/usr/bin/env bash
# Kumru-2B GGUF hattı: llama.cpp'ye Kumru ön işleyicisi (llama_kumru_patch.py), derleme, HF'e karşı tokenizasyon
# denetimi; eşleşirse run_model.sh (quant ve Vikipedi KLD) ve flores_kld.sh. Derleme çalışan ikili dosyaları
# değiştirdiği için önce imatrix_seeds.sh'in bitmesini bekler. Kullanım: bash kumru_pipeline.sh
set -u -o pipefail
WORK=$HOME/work/tr-quant
M=$WORK/models/kumru-2b
R=$WORK/results/kumru-2b
PY=$WORK/.venv-convert/bin/python
BIN=$WORK/llama.cpp/build/bin
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

until grep -q '\] bitti' "$WORK/imatrix_seeds.log" 2>/dev/null; do sleep 60; done
log "imatrix_seeds bitti"

id=$(~/bin/lease wait --owner tr-quant --gpus none --mode shared --hours 1 \
  --note 'llama.cpp Kumru ön işleyici yaması ve derleme (CPU 96-191)' --timeout-min 240 | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "derleme kiralaması alınamadı"; exit 1; }
log "derleme kiralaması: $id"
python3 "$WORK/llama_kumru_patch.py" "$WORK/llama.cpp" || { ~/bin/lease release "$id"; exit 1; }
git -C "$WORK/llama.cpp" diff >"$R/llama-cpp-kumru.patch"
if ! taskset -c 96-191 cmake --build "$WORK/llama.cpp/build" -j 96 \
  --target llama-imatrix llama-quantize llama-perplexity llama-tokenize >"$R/build.log" 2>&1; then
  log "derleme hatası ($R/build.log)"
  ~/bin/lease release "$id"
  exit 1
fi
log "derleme tamam"
log "dönüştürme"
taskset -c 96-191 "$PY" "$WORK/llama.cpp/convert_hf_to_gguf.py" "$M/hf" --outtype bf16 --outfile "$M/bf16.gguf" \
  >"$R/convert.log" 2>&1 || { log "dönüştürme hatası"; ~/bin/lease release "$id"; exit 1; }
"$PY" "$WORK/check_vocab.py" "$M/bf16.gguf" "$M/hf/tokenizer.json" | tee "$R/check_vocab.log" ||
  { log "vocab uyuşmazlığı"; ~/bin/lease release "$id"; exit 1; }
log "tokenizasyon denetimi"
# Kaynak kod örnekleri llama.cpp b12116 ağacından (llama-vocab.cpp yamalı hâli): kod, Unicode tablosu, Python
cp "$WORK/llama.cpp/src/llama-vocab.cpp" /tmp/kumru_code_sample.txt
cp "$WORK/llama.cpp/src/unicode.cpp" /tmp/kumru_code_unicode.txt
cd "$WORK"
if ! "$PY" kumru_tokenize_check.py "$BIN/llama-tokenize" "$M/bf16.gguf" "$M/hf/tokenizer.json" \
  text/flores_tr.txt text/flores_en.txt text/calib_tr.txt text/eval_en.txt \
  /tmp/kumru_code_sample.txt llama.cpp/conversion/base.py /tmp/kumru_code_unicode.txt \
  | tee "$R/tokenize_check_noescape.log"; then
  log "tokenizasyon HF'ten farklı; quant yok"
  ~/bin/lease release "$id"
  exit 1
fi
~/bin/lease release "$id" && log "derleme kiralaması bırakıldı: $id"

gid=$(~/bin/lease wait --owner tr-quant --gpus 2,3,4,5 --mode shared --hours 2 \
  --note 'Kumru-2B quant ve KLD (Vikipedi, FLORES+)' --timeout-min 240 | awk 'NR==1 {print $1}')
[ -n "$gid" ] || { log "GPU kiralaması alınamadı"; exit 1; }
log "GPU kiralaması: $gid (GPU 2,3,4,5)"
GPU_LIST=2,3,4,5 CPUSET=96-191 bash "$WORK/run_model.sh" vngrs/Kumru-2B 0b3cf37f657ef4013bdc99598545f5d317bd7397 \
  kumru-2b >"$WORK/run_kumru-2b.log" 2>&1 || log "HATA run_model (run_kumru-2b.log)"
GPU_LIST=2,3,4,5 CPUSET=96-191 bash "$WORK/flores_kld.sh" kumru-2b >>"$WORK/run_kumru-2b.log" 2>&1 ||
  log "HATA flores_kld"
~/bin/lease release "$gid" && log "GPU kiralaması bırakıldı: $gid"
log "bitti"
