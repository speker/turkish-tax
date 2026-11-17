#!/usr/bin/env bash
# 16-17 Kasım takip ölçümlerinin komutları (makalede "Follow-up measurements"). Her oturum bench.sh'ı yayımlanan
# koşulardaki ayarlarla çağırır; çıktı dizinleri results/takip-2026-11-17/kosu/<oturum>/ ve
# results/tekrar-2026-11-17/kosu/<oturum>/ ile aynı adı taşır. Koşular ana makinede (8×H200) ve ikinci makinede
# (5×H200) yapıldı; GPU ayırma, makinedeki başka işleri durdurup geri açma ve izleme adımları burada yok.
# Aynı oturumda & ile başlayan hatlar aynı anda koştu (her hat ayrı GPU, CPU dilimi ve port).
# Kullanım: bash takip_kosulari.sh OTURUM
#   OTURUM: takip takip-tek takip-istemci takip-ek takip-ek2 takip-ek3 tp2ara yedek dogal fp8-belge kv
set -u
WORK=$HOME/work/tr-serve
M=$HOME/models
TG=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf
Q=$M/qwen3.8-27b
MI=$M/ministral-3-14b-bf16
G4=$M/gemma-4-12b-it
PY=${PY:-python3}
cd "$WORK" || exit 1
export ORDER=alt

run() {
  local name=$1 model=$2 tp=$3 gpus=$4 cpus=$5 port=$6
  CPUSET=$cpus PORT=$port bash bench.sh "$name" "$model" "$tp" "$gpus" --no-enable-prefix-caching \
    > "bench_$name.log" 2>&1 && echo "tamam $name" || echo "HATA $name"
}

case ${1:-} in
takip)
  export RESULTS_ROOT=$WORK/results-takip
  # Qwen3.8-27B tensör paralelliği ara noktaları ve uzun girdi (Türkçe çıktı 256 ve 360 token, İngilizce 256)
  REPS="1 2 3" CONC="64:1024 96:1536 128:2048" run qwen3.8-27b-tp1 "$Q" 1 0 0-31 18121 &
  REPS="1 2 3" CONC="64:1024 96:1536 128:2048 160:2560" run qwen3.8-27b-tp2 "$Q" 2 1,2 32-95 18122 &
  REPS="1 2" CONC="1:16 16:128 64:512 128:1024" LANGS="trlong trlong:360 enlong" \
    run qwen3.8-27b-uzun "$Q" 1 3 96-127 18123 &
  REPS="1 2 3" CONC="64:1024 96:1536 128:2048 256:4096" run qwen3.8-27b-tp4 "$Q" 4 4,5,6,7 128-191 18124 &
  wait
  # Gemma 4 uzun girdi, Qwen uzun girdide eşit token (İngilizce istem Türkçe token sayısına doldurulur),
  # Qwen açık döngü 4/8/12 istek/sn (istek başına gecikme kaydıyla). GPU 1-2'de aynı anda doğal bitiş sunucuları.
  $PY make_padded_prompts.py data/prompts --kume=long "qwen=$Q/tokenizer.json"
  REPS="1 2" CONC="1:16 16:128 64:512 128:1024" LANGS="trlong trlong:350 enlong" \
    run gemma-4-12b-it-uzun "$G4" 1 0 0-31 18121 &
  REPS="1 2" CONC="1:16 16:128 64:512 128:1024" LANGS="trlong enlong enlongpad-qwen" \
    run qwen3.8-27b-uzun-esit "$Q" 1 7 64-95 18127 &
  for spec in 4:4:128-159:18124 8:5:160-191:18125 12:3:96-127:18123; do
    IFS=: read -r rate gpu cpus port <<< "$spec"
    DETAILED=1 RATE=$rate CONC="1024:$((rate * 150))" REPS="1 2" \
      run "qwen3.8-27b-acik-r$rate" "$Q" 1 "$gpu" "$cpus" "$port" &
  done
  wait
  ;;
takip-tek)
  # Komşusuz: her koşu makinede tek başına
  export RESULTS_ROOT=$WORK/results-takip-tek
  REPS="1 2 3 4 5" CONC="128:2048" run qwen3.8-27b-tp2 "$Q" 2 1,2 32-95 18122
  REPS="1 2" CONC="128:2048" run qwen3.8-27b-tp4 "$Q" 4 4,5,6,7 128-191 18124
  REPS="1 2" CONC="128:2048" run qwen3.8-27b-tp1 "$Q" 1 0 0-31 18121
  REPS="1 2 3 4 5" CONC="256:4096" run ministral-3-14b-bf16 "$MI" 1 0 0-31 18121
  ;;
takip-istemci)
  # Benchmark istemcisi sunucuyla aynı 32 vCPU'da ve ayrı 32 vCPU'da
  export RESULTS_ROOT=$WORK/results-takip-istemci
  REPS="1 2" CONC="64:1024 256:4096" run turkish-gemma-9b-t1 "$TG" 1 0 0-31 18121
  REPS="1 2" CONC="64:1024 256:4096" CLIENT_CPUSET=32-63 run turkish-gemma-9b-t1-ayri "$TG" 1 0 0-31 18121
  ;;
takip-ek)
  # GPU 2-7'de başka vLLM sunucuları çalışırken. Ministral ölçekli çıktı: Türkçe 380 token = 256 × token oranı.
  # Qwen açık döngü, içerik eşleşmeli: Türkçe 365 token.
  export RESULTS_ROOT=$WORK/results-takip-ek
  CONC="256:4096" REPS="1 2 3 4 5" LANGS="tr:380 en" \
    run ministral-3-14b-bf16-olcekli "$MI" 1 0 0-31 18131 &
  {
    for rate in 8 12; do
      DETAILED=1 RATE=$rate CONC="1024:$((rate * 150))" REPS="1 2" LANGS="tr:365 en" \
        run "qwen3.8-27b-acik-icerik-r$rate" "$Q" 1 1 32-63 18132
    done
  } &
  wait
  ;;
takip-ek2)
  # GPU 2-7'de başka vLLM sunucuları çalışırken. Turkish-Gemma uzun girdi (Türkçe 355 token = 256 × uzun istemlerin
  # token oranı 1,388); Qwen uzun girdi 256 istekte, KV önbellek dolu.
  export RESULTS_ROOT=$WORK/results-takip-ek2
  REPS="1 2" CONC="1:16 16:128 64:512 128:1024" LANGS="trlong trlong:355 enlong" \
    run turkish-gemma-9b-t1-uzun "$TG" 1 0 0-31 18141 &
  REPS="1 2" CONC="256:1024" LANGS="trlong:360 enlong" run qwen3.8-27b-uzun-kv "$Q" 1 1 32-63 18142 &
  wait
  ;;
takip-ek3)
  # GPU 1-7'de başka vLLM sunucuları çalışırken. Turkish-Gemma uzun girdide eşit token kontrolü.
  export RESULTS_ROOT=$WORK/results-takip-ek3
  $PY make_padded_prompts.py data/prompts --kume=long "tg=$TG/tokenizer.json"
  REPS="1 2" CONC="16:128 64:512 128:1024" LANGS="trlong:355 enlongpad-tg:355 enlong" \
    run turkish-gemma-9b-t1-uzun-esit "$TG" 1 0 0-31 18151
  ;;
tp2ara)
  # Makaledeki iki GPU'lu Qwen ölçümüyle aynı GPU (1,2) ve CPU (0-63) dilimi; makine boşken
  export RESULTS_ROOT=$WORK/results-tp2ara REPS="1 2" CONC="32:512 64:1024 128:2048 256:4096 384:6144"
  run qwen3.8-27b-tp2-ara "$Q" 2 1,2 0-63 18120
  ;;
yedek)
  # İkinci makine (5×H200, NVSwitch), aynı imaj, model dosyaları ve istemler; makinede başka GPU işi yokken
  export RESULTS_ROOT=$WORK/results-yedek
  python3 make_long_prompts.py --docs data/prompts/docs.json --out data/prompts
  REPS="1 2" run turkish-gemma-9b-t1-tek "$TG" 1 0 0-31 18101
  {
    REPS="1 2 3 4 5" run turkish-gemma-9b-t1 "$TG" 1 0 0-31 18101
    BENCH_ARGS="--temperature 0" CONC="64:1024 256:4096" REPS="1 2" run turkish-gemma-9b-t1-t0 "$TG" 1 0 0-31 18101
    IGNORE_EOS=0 DETAILED=1 CONC="256:2048" LANGS="tr:1024 en:1024" REPS="1 2" \
      run turkish-gemma-9b-t1-dogal "$TG" 1 0 0-31 18101
    for rate in 10 20; do
      IGNORE_EOS=0 DETAILED=1 RATE=$rate CONC="1024:2048" LANGS="tr:1024 en:1024" REPS="1 2" \
        run "turkish-gemma-9b-t1-acik-r$rate" "$TG" 1 0 0-31 18101
    done
  } &
  {
    REPS="1 2 3 4 5" run gemma-4-12b-it "$G4" 1 1 32-63 18102
    CONC="1:16 16:128 64:512 128:1024" LANGS="trlong enlong" REPS="1 2" \
      run qwen3.8-27b-uzun-girdi "$Q" 1 1 32-63 18102
  } &
  REPS="1 2 3 4 5" run qwen3.8-27b "$Q" 1 3 96-127 18104 &
  {
    REPS="1 2 3 4 5" run ministral-3-14b-bf16 "$MI" 1 2 64-95 18103 &
    CONC="1:128 16:1024 64:4096 256:16384" REPS="1 2 3 4 5" run kumru-2b-uzun "$M/kumru-2b" 1 4 128-159 18105 &
    wait
    CONC="32:512 64:1024 128:2048 256:4096 384:6144" REPS="1 2" \
      run qwen3.8-27b-tp2-ara "$Q" 2 2,4 64-95,128-159 18106
  } &
  wait
  ;;
dogal)
  # Sonuna kadar üretilen cevaplar (düşünme kapalı, en çok 3.500 token, dil başına 562 istem, iki tohum).
  # Qwen3.8-27B: aynı vLLM 0.33.0 resmî imajında FP8 sunucular (--quantization fp8 --max-model-len 8192
  # --gpu-memory-utilization 0.92 --max-num-seqs 512), sekiz GPU'da birer örnek, portlar 8301-8308.
  bash dogal_tek.sh gemma-4-12b-it google/gemma-4-12B-it 1 32-47 18131 gemma-4-12b-it
  bash dogal_tek.sh ministral-3-14b-bf16 mistralai/Ministral-3-14B-Instruct-2512 2 48-63 18133 \
    ministral-3-14b-bf16-t015 --sablonsuz --temperature 0.15
  $PY dogal_cevap.py --ports 8301,8302,8303,8304,8305,8306,8307,8308 --model qwen/qwen3.8-27b \
    --token-orani token_ratio.csv --out results-dogal/qwen3.8-27b-fp8
  ;;
fp8-belge)
  # Belge düzeyinde FP8 (562 belge) ve Turkish-Gemma'da parça düzeyinin ikinci koşusu
  bash fp8_belge.sh turkish-gemma-9b-t1:0:0-15 gemma-4-12b-it:1:16-31 qwen3.8-27b:2:32-47 kumru-2b:3:48-63 \
    ministral-3-14b-bf16:4:64-79 ministral-fp8-yerel:5:80-95
  OUT=$WORK/results/fp8-kalite-parca-tekrar LEASE_GPUS=0,1 ONLY=turkish-gemma-9b-t1 bash fp8_quality.sh
  ;;
kv)
  # results/takip-2026-11-17/kv/: sunucu logundaki KV önbellek satırları
  for p in results-takip/qwen3.8-27b-uzun results-takip/gemma-4-12b-it-uzun results-takip-ek2/qwen3.8-27b-uzun-kv \
      results-takip-ek2/turkish-gemma-9b-t1-uzun results-takip-ek3/turkish-gemma-9b-t1-uzun-esit; do
    root=$(dirname "$p") n=$(basename "$p")
    mkdir -p "kv/${root#results-}/$n"
    grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs, GPU KV cache usage: [0-9.]+%" "$p/server.log" \
      > "kv/${root#results-}/$n/kv.log"
    grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" "$p/server.log" >> "kv/${root#results-}/$n/kv.log"
  done
  ;;
*)
  sed -n '2,8p' "$0"
  exit 1
  ;;
esac
