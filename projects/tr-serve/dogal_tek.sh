#!/usr/bin/env bash
# Tek modelde doğal bitiş (takip_kosulari.sh'taki dogal ile aynı sunucu ayarı).
# Kullanım: bash dogal_tek.sh AD REPO GPU CPU PORT CIKTI_ADI [dogal_cevap.py argümanları...]
#   örnek: bash dogal_tek.sh ministral-3-14b-bf16 mistralai/Ministral-3-14B-Instruct-2512 2 48-63 18133 \
#            ministral-3-14b-bf16-t015 --sablonsuz --temperature 0.15
set -u
name=$1 repo=$2 gpu=$3 cpus=$4 port=$5 out=$6
shift 6
WORK=$HOME/work/tr-serve
c=tr-serve-dogal-$out
cd "$WORK" || exit 1
sudo docker rm -f "$c" > /dev/null 2>&1
sudo docker run -d --name "$c" --runtime=nvidia --gpus "\"device=$gpu\"" --ipc=host --cpuset-cpus "$cpus" \
  -p "127.0.0.1:$port:8000" -v "$HOME/models/$name:/model:ro" tr-serve-vllm:0.33.0-bench \
  --model /model --served-model-name "$name" --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --seed 0 > /dev/null
for _ in $(seq 1 120); do curl -sf "127.0.0.1:$port/health" > /dev/null && break; sleep 10; done
mkdir -p results-dogal
~/work/tr-decide/.venv-pusula/bin/python dogal_cevap.py --ports "$port" --model "$name" --repo "$repo" \
  --token-orani token_ratio.csv --out "results-dogal/$out" "$@" > "results-dogal/$out.log" 2>&1
rc=$?
sudo docker logs "$c" > "results-dogal/$out.server.log" 2>&1
sudo docker rm -f "$c" > /dev/null
echo "$(date +%T) doğal bitiş $out çıkış $rc"
