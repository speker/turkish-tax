#!/usr/bin/env bash
# v3: makaledeki servis tablosunun prefix caching KAPALI tekrarı (v2'de vLLM 0.33'ün varsayılanı olan prefix caching
# açıktı; isabet Qwen3.8'de %0, diğerlerinde %75-96). Sonuçlar results-v3/ altına, v2'ye dokunmadan.
# GPU'lar paralel kullanılır, önce etkilenme kontrolü:
#   1. Kontrol: Turkish-Gemma BF16, GPU 1, CPU 0-31, tek başına (turkish-gemma-9b-t1-tek).
#   2. Paralel parti: GPU 1-5, her hat ayrı port ve 32 vCPU; GPU 1'de aynı Turkish-Gemma tekrar.
#   3. İki Turkish-Gemma ölçümü her eşzamanlılık ve dilde %3 içindeyse paralel parti geçerli; değilse paralel partinin
#      modelleri GPU 1'de tek tek yeniden ölçülür (results-v3/seri/).
#   4. Tek başına: gpt-oss-120b, Kumru-2B. 5. Qwen3.8-27B TP=2 ve TP=4.
# VM altı GPU'nun hepsini NUMA 0'a (CPU 0-47) bağlı gösteriyor; fiziksel yerleşim bilinmediği için hatlara ayrık
# 32 vCPU blokları verilir. Throughput ölçümü bütün makineyi ister: quiet kiralama ve boş GPU beklenir.
set -u
WORK=$HOME/work/tr-serve
export RESULTS_ROOT=$WORK/results-v3
mkdir -p "$RESULTS_ROOT"
TG=$HOME/work/tr-quant/models/turkish-gemma-9b-t1/hf
M=$HOME/models
THRESHOLD=3
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }

night() {
  local h
  # HEMEN=1: gece penceresini beklemeden başlat
  [ -n "${HEMEN:-}" ] && return 0
  h=$(date +%-H)
  [ "$h" -ge 22 ] || [ "$h" -lt 2 ]
}
gpus_idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
log "başlangıç penceresi 22:00-01:59, quiet kiralama ve boş GPU bekleniyor"
# lease, GPU'da kiralamasız süreç varken de quiet verir; kiralamadan sonra GPU'ların boşalması ayrıca beklenir.
id=
while :; do
  until night; do sleep 300; done
  if out=$(~/bin/lease wait --owner tr-serve --gpus all --mode quiet --hours 6 --timeout-min 30 \
    --note 'servis tablosu prefix caching kapalı, paralel (~4 sa)'); then
    id=$(awk 'NR==1 {print $1}' <<<"$out")
  fi
  if [ -n "$id" ]; then
    for _ in $(seq 1 20); do gpus_idle && break; sleep 30; done
    gpus_idle && break
    log "quiet alındı ama GPU'da süreç kaldı; bırakılıp yeniden denenecek"
    ~/bin/lease release "$id" >/dev/null
    id=
  fi
  sleep 60
done
log "quiet kiralama: $id"

# run AD MODEL_DIZINI TP GPU'LAR CPUSET PORT [vLLM argümanları...]
run() {
  local name=$1 dir=$2 tp=$3 gpus=$4 cpus=$5 port=$6
  shift 6
  log "başla $name (GPU $gpus, CPU $cpus)"
  if CPUSET=$cpus PORT=$port bash "$WORK/bench.sh" "$name" "$dir" "$tp" "$gpus" --no-enable-prefix-caching "$@" \
    >"$WORK/bench_v3_$name.log" 2>&1; then
    log "tamam $name"
  else
    log "HATA $name"
  fi
}

# 1. Kontrol
RESULTS_ROOT=$WORK/results-v3 run turkish-gemma-9b-t1-tek "$TG" 1 1 0-31 8101
if ! grep -q 'enable_prefix_caching=False' "$RESULTS_ROOT/turkish-gemma-9b-t1-tek/server.log" 2>/dev/null; then
  log "HATA: server.log'da enable_prefix_caching=False yok; v3 durdu"
  ~/bin/lease release "$id"
  exit 1
fi

# 2. Paralel parti
log "paralel parti"
lane1() {
  run turkish-gemma-9b-t1 "$TG" 1 1 0-31 8101
  run turkish-gemma-9b-t1-fp8 "$TG" 1 1 0-31 8101 --quantization fp8
  run ministral-3-14b "$M/ministral-3-14b" 1 1 0-31 8101
}
lane2() {
  run gemma-4-12b-it "$M/gemma-4-12b-it" 1 2 32-63 8102
  run gemma-4-12b-it-fp8 "$M/gemma-4-12b-it" 1 2 32-63 8102 --quantization fp8
}
lane3() {
  run ministral-3-14b-bf16 "$M/ministral-3-14b-bf16" 1 3 64-95 8103
  run ministral-3-14b-bf16-fp8 "$M/ministral-3-14b-bf16" 1 3 64-95 8103 --quantization fp8
}
lane4() { run qwen3.8-27b "$M/qwen3.8-27b" 1 4 96-127 8104; }
lane5() { run qwen3.8-27b-fp8 "$M/qwen3.8-27b" 1 5 128-159 8105 --quantization fp8; }
lane1 &
lane2 &
lane3 &
lane4 &
lane5 &
wait
log "paralel parti bitti"

# 3. Etkilenme kararı
verdict=$(python3 - "$RESULTS_ROOT" "$THRESHOLD" <<'EOF'
import json, pathlib, statistics, sys
root, thr = pathlib.Path(sys.argv[1]), float(sys.argv[2])
def med(d):
    out = {}
    for f in d.glob("*_r*.json"):
        lang, c, _ = f.stem.split("_")
        out.setdefault((lang, c), []).append(json.loads(f.read_text())["output_throughput"])
    return {k: statistics.median(v) for k, v in out.items()}
a, b = med(root / "turkish-gemma-9b-t1-tek"), med(root / "turkish-gemma-9b-t1")
diffs = {f"{k[0]} {k[1]}": (b[k] / a[k] - 1) * 100 for k in a if k in b}
worst = max(abs(v) for v in diffs.values()) if diffs else 999
(root / "etkilenme_kontrolu.json").write_text(json.dumps({"esik_yuzde": thr, "fark_yuzde": diffs, "en_buyuk": worst}, indent=1))
print("gecerli" if diffs and worst <= thr else "gecersiz", f"{worst:.2f}")
EOF
)
log "etkilenme kontrolü: $verdict (eşik %$THRESHOLD)"
if [[ $verdict != gecerli* ]]; then
  log "paralel parti geçersiz; modeller GPU 1'de tek tek yeniden ölçülüyor"
  export RESULTS_ROOT=$WORK/results-v3/seri
  mkdir -p "$RESULTS_ROOT"
  run turkish-gemma-9b-t1-fp8 "$TG" 1 1 0-31 8101 --quantization fp8
  run ministral-3-14b "$M/ministral-3-14b" 1 1 0-31 8101
  run gemma-4-12b-it "$M/gemma-4-12b-it" 1 1 0-31 8101
  run gemma-4-12b-it-fp8 "$M/gemma-4-12b-it" 1 1 0-31 8101 --quantization fp8
  run ministral-3-14b-bf16 "$M/ministral-3-14b-bf16" 1 1 0-31 8101
  run ministral-3-14b-bf16-fp8 "$M/ministral-3-14b-bf16" 1 1 0-31 8101 --quantization fp8
  run qwen3.8-27b "$M/qwen3.8-27b" 1 1 0-31 8101
  run qwen3.8-27b-fp8 "$M/qwen3.8-27b" 1 1 0-31 8101 --quantization fp8
  export RESULTS_ROOT=$WORK/results-v3
fi

# 4. Tek başına
run gpt-oss-120b "$M/gpt-oss-120b" 1 1 0-31 8101
CONC="1:128 16:1024 64:4096 256:16384" REPS="1 2 3" run kumru-2b-uzun "$M/kumru-2b" 1 1 0-31 8101

# 5. Tensor parallel
run qwen3.8-27b-tp2 "$M/qwen3.8-27b" 2 1,2 0-63 8101
run qwen3.8-27b-tp4 "$M/qwen3.8-27b" 4 1,2,3,4 0-127 8101

~/bin/lease release "$id" && log "kiralama bırakıldı: $id"
log "bitti"
