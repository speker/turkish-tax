#!/usr/bin/env bash
# Qwen3.8-27B tensor parallel, yüksek eşzamanlılık (v3 ayarları, prefix caching kapalı). v3'te TP1/2/4 yalnız
# 256 eşzamanlılıkta karşılaştırıldı; TP2/TP4'te KV cache boş kaldığı için GPU başına verim haksız düşük
# görünebilir. vLLM 0.33 sunucu varsayılanı max_num_seqs=1024. TP1 (GPU 1) ve TP2 (GPU 2-3) aynı anda, sonra TP4.
set -u
WORK=$HOME/work/tr-serve
export RESULTS_ROOT=$WORK/results-v3
M=$HOME/models/qwen3.8-27b
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
id=$(~/bin/lease wait --owner tr-serve --gpus all --mode quiet --hours 4 --timeout-min 30 \
  --note 'Qwen3.8 TP yüksek eşzamanlılık (~2 sa)' | awk 'NR==1 {print $1}')
[ -n "$id" ] || { log "kiralama alınamadı"; exit 1; }
log "quiet kiralama: $id"
run() {
  local name=$1 tp=$2 gpus=$3 cpus=$4 port=$5 conc=$6
  log "başla $name (TP=$tp, GPU $gpus, CPU $cpus, $conc)"
  if CONC=$conc CPUSET=$cpus PORT=$port bash "$WORK/bench.sh" "$name" "$M" "$tp" "$gpus" --no-enable-prefix-caching \
    >"$WORK/bench_v3_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
}
# İlk koşuda TP2 ve TP4'e 1024:16384 da verilmişti; istemci dosya sınırına takıldı (*/gecersiz_ulimit/).
# Geçerli 1.024 koşusu tp1024_v3.sh.
run qwen3.8-27b-yuksek 1 1 0-31 8101 "256:4096 512:8192" &
run qwen3.8-27b-tp2-yuksek 2 2,3 32-95 8102 "256:4096 512:8192" &
wait
run qwen3.8-27b-tp4-yuksek 4 1,2,3,4 0-127 8101 "256:4096 512:8192"
~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
