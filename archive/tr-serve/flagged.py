"""İki koşusu arasında eşikten fazla fark olan yapılandırmaları listeler (tekrar koşusu için).

Kullanım: python3 flagged.py SONUÇ_DIZINI [--esik 5] [--haric a,b]
Çıktı satırı: <model_klasörü> <dil> <eşzamanlılık>:<istem_sayısı>
Varsayılan hariç: Kumru-2B (CPU sınırında; kumru_long.sh ile ayrı ölçülüyor) ve ministral-3-14b-fp8
(repo zaten FP8, bu koşu tablodan çıktı).
"""

import argparse
import json
import re
from pathlib import Path

RUN = re.compile(r"^(tr|en)_c(\d+)_r(\d+)\.json$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--esik", type=float, default=5.0)
    parser.add_argument("--haric", default="kumru-2b,kumru-2b-fp8,ministral-3-14b-fp8")
    args = parser.parse_args()
    skip = set(args.haric.split(","))
    for model_dir in sorted(p for p in args.results.iterdir() if p.is_dir() and p.name not in skip):
        runs: dict[tuple, list] = {}
        for f in model_dir.glob("*.json"):
            m = RUN.match(f.name)
            if m:
                d = json.loads(f.read_text(encoding="utf-8"))
                runs.setdefault((m[1], int(m[2])), []).append((d["output_throughput"], d["num_prompts"]))
        for (lang, conc), vals in sorted(runs.items()):
            tput = [v[0] for v in vals]
            if len(tput) < 2:
                continue
            mid = sorted(tput)[len(tput) // 2]
            if (max(tput) - min(tput)) / mid * 100 > args.esik:
                print(f"{model_dir.name} {lang} {conc}:{vals[0][1]}")


if __name__ == "__main__":
    main()
