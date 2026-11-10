#!/usr/bin/env bash
# İki Gemma'nın TPOT eğimi farkında çekirdek ile mimariyi ayırma (Post yorumu, 10 Kasım): İngilizce istem, çıktı 256
# ve N = 256 × token oranı, 256 eşzamanlılık, beş tekrar.
# vLLM 0.33.0'da Gemma 4'ün seçtiği TRITON_FLASH_ATTN bileşik bir arka uç: görüntü maskeli sorgular Triton'a, metin
# (nedensel) sorguları FlashAttention'a gider; 512 genişliğindeki baş FlashAttention'da yalnız FA4 ile çalışır. Bu
# yüzden metin koşularında fark FA4 ile FA3 arasında. Hücreler:
#   turkish-gemma-9b-t1         varsayılan (FLASH_ATTN, FA3)            GPU 1, v3 ile aynı yerleşim
#   gemma-4-12b-it              varsayılan (TRITON_FLASH_ATTN, FA4)     GPU 2, v3 ile aynı yerleşim
#   turkish-gemma-9b-t1-fa4     FLASH_ATTN, FA4 zorlanmış               GPU 3
#   turkish-gemma-9b-t1-triton  TRITON_ATTN                             GPU 4
#   gemma-4-12b-it-triton       TRITON_ATTN                             GPU 5
# vLLM bir hücreyi kabul etmezse o hücre HATA yazar, diğerleri sürer. Seçilen arka uç server.log'dan backends.txt'ye.
set -u
WORK=$HOME/work/tr-serve
export RESULTS_ROOT=$WORK/results-cekirdek
mkdir -p "$RESULTS_ROOT"
TG=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf
G4=$HOME/models/gemma-4-12b-it
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
gpus_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }

id=$(~/bin/lease wait --owner tr-serve --gpus all --mode quiet --hours 2 --timeout-min "${WAIT_MIN:-900}" \
  --note 'çekirdek/mimari ayrımı: 2 model, 5 hücre, 256 eşzamanlılık, 5 tekrar (~45 dk)' | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "quiet kiralama alınamadı"; exit 1; }
for _ in $(seq 1 20); do gpus_idle && break; sleep 30; done
gpus_idle || { log "GPU'da süreç kaldı; bırakıldı"; ~/bin/lease release "$id"; exit 1; }
log "quiet kiralama: $id"

# run AD MODEL_DIZINI GPU CPUSET PORT N [ek vLLM argümanları]
run() {
  local name=$1 dir=$2 gpu=$3 cpus=$4 port=$5 n=$6
  shift 6
  log "başla $name (GPU $gpu, CPU $cpus, N $n, $*)"
  if CONC="256:4096" REPS="1 2 3 4 5" LANGS="en:$n en" CPUSET=$cpus PORT=$port \
    bash "$WORK/bench.sh" "$name" "$dir" 1 "$gpu" --no-enable-prefix-caching "$@" >"$WORK/bench_cekirdek_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
}
run turkish-gemma-9b-t1 "$TG" 1 0-31 8111 357 &
run gemma-4-12b-it "$G4" 2 32-63 8112 349 &
run turkish-gemma-9b-t1-fa4 "$TG" 3 64-95 8113 357 --attention-config.flash_attn_version=4 &
run turkish-gemma-9b-t1-triton "$TG" 4 96-127 8114 357 --attention-backend TRITON_ATTN &
run gemma-4-12b-it-triton "$G4" 5 128-159 8115 349 --attention-backend TRITON_ATTN &
wait
for d in "$RESULTS_ROOT"/*/; do
  grep -h -E 'attention backend out of|FlashAttention version' "$d/server.log" 2>/dev/null |
    sed -E "s|^.*\] |$(basename "$d"): |"
done >"$RESULTS_ROOT/backends.txt"
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
