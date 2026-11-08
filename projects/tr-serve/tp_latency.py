"""Qwen3.8-27B: TP düzenlerinde gecikme ve 1M Türkçe kelime başına GPU-saat, aynı tabloda.

Azami hızda TP kelime başına pahalı; gecikme hedefi konunca sıra değişebilir. Bu betik iki karşılaştırmayı da
ozet.csv'den üretir. Hız ve gecikme yüzdelikleri koşular üzerinden medyan (summarize.py).

Kullanım: uv run python projects/tr-serve/tp_latency.py [results/serve-v3/ozet.csv]
"""

import csv
import sys
from pathlib import Path

MODELS = {
    "qwen3.8-27b",
    "qwen3.8-27b-yuksek",
    "qwen3.8-27b-tp2",
    "qwen3.8-27b-tp2-yuksek",
    "qwen3.8-27b-tp2-1024",
    "qwen3.8-27b-tp4",
    "qwen3.8-27b-tp4-yuksek",
    "qwen3.8-27b-tp4-1024",
}


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r["model"] in MODELS and r["dtype"] == "bf16" and r["dil"] == "tr"
        ]
    for r in rows:
        r["tp"] = int(r["tp"])
        r["c"] = int(r["eszamanlilik"])
    return sorted(rows, key=lambda r: (r["tp"], r["c"], r["model"]))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results/serve-v3/ozet.csv"
    print("TP  c     kaynak                    TR tok/s  TPOT p50  TTFT p50  TTFT p99  GPU-h/1M  fark%")
    for r in load(path):
        print(
            f"{r['tp']:<3} {r['c']:<5} {r['model']:<25} {float(r['cikti_tok_s']):8.0f}"
            f"  {float(r['tpot_p50_ms']):7.1f}  {float(r['ttft_p50_ms']):8.0f}"
            f"  {float(r['ttft_p99_ms']):8.0f}  {float(r['gpu_saat_1m_kelime']):8.3f}"
            f"  {r['kosu_farki_yuzde']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
