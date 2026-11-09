"""Dolgulu istem sınamasının özeti (padded_v3.sh): her model için dört koşu, 256 eşzamanlılık.

Kullanım: python padded_summary.py [SONUÇ_DIZINI]   (varsayılan results/dolgu)
Oranlar EN tok/s ÷ X tok/s biçiminde (1'den büyükse X daha yavaş). Bağlam uzunluğu kalan farkı açıklıyorsa
"dolgulu EN ÷ TR" 1'e yakın çıkar; "EN 256 ÷ TR" kalan yük çarpanıdır (night_scaled_v3.sh ile aynı tanım).
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "dolgu"
MODELS = {
    "turkish-gemma-9b-t1": (357, "tg"),
    "gemma-4-12b-it": (349, "g4"),
    "ministral-3-14b-bf16": (380, "mi"),
    "qwen3.8-27b": (365, "qw"),
}


def runs(d: Path, label: str) -> list[dict]:
    js = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob(f"{label}_c256_r*.json"))]
    assert js and all(j["failed"] == 0 and j["completed"] == 4096 for j in js), (d.name, label)
    return js


def med(js: list[dict], key: str) -> float:
    return statistics.median(j[key] for j in js)


def spread(js: list[dict]) -> float:
    x = [j["output_throughput"] for j in js]
    return (max(x) - min(x)) / statistics.median(x) * 100


def main() -> int:
    print(
        "model | tok/s TR / dolgulu EN / EN N / EN 256 | EN256÷TR | dolguluEN÷TR | EN N÷dolguluEN "
        "| EN256÷EN N | istem token TR / dolgulu EN / EN | TPOT ms TR / dolgulu EN / EN N / EN 256 "
        "| koşu farkı en çok %"
    )
    for name, (n, pad) in MODELS.items():
        d = ROOT / name
        tr, ep, en_n, en = (
            runs(d, f"tr-o{n}"),
            runs(d, f"enpad-{pad}-o{n}"),
            runs(d, f"en-o{n}"),
            runs(d, "en"),
        )
        t = {k: med(v, "output_throughput") for k, v in (("tr", tr), ("ep", ep), ("en_n", en_n), ("en", en))}
        inp = [med(v, "total_input_tokens") / 4096 for v in (tr, ep, en)]
        tpot = [med(v, "median_tpot_ms") for v in (tr, ep, en_n, en)]
        worst = max(spread(v) for v in (tr, ep, en_n, en))
        print(
            f"{name} | {t['tr']:,.0f} / {t['ep']:,.0f} / {t['en_n']:,.0f} / {t['en']:,.0f} | "
            f"{t['en'] / t['tr']:.3f} | {t['ep'] / t['tr']:.3f} | {t['en_n'] / t['ep']:.3f} | "
            f"{t['en'] / t['en_n']:.3f} | {inp[0]:.1f} / {inp[1]:.1f} / {inp[2]:.1f} | "
            f"{' / '.join(f'{x:.1f}' for x in tpot)} | {worst:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
