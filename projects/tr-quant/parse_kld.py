"""llama-perplexity KLD log'larını tabloya döker.

Kullanım: python parse_kld.py SONUÇ_DIZINI [--out kld.csv]
Girdi: kld_<varyant>_<dil>.log, ppl_bf16_<dil>.log ve sizes.tsv. Çıktı: varyant ve dil başına bir satır.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

PATTERNS = {
    "ppl_q": r"Mean PPL\(Q\)\s*:\s*([\d.]+)",
    "ppl_base": r"Mean PPL\(base\)\s*:\s*([\d.]+)",
    "kld_mean": r"Mean\s+KLD:\s*([\d.]+)",
    "kld_median": r"Median\s+KLD:\s*([\d.]+)",
    "kld_p99": r"99\.0%\s+KLD:\s*([\d.]+)",
    "same_top_p": r"Same top p:\s*([\d.]+)",
}
FIELDS = ["varyant", "dil", "boyut_gb", *PATTERNS]


def parse(text: str) -> dict:
    out = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, text)
        out[key] = float(m.group(1)) if m else None
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sizes = {}
    sizes_file = args.results / "sizes.tsv"
    if sizes_file.exists():
        for line in sizes_file.read_text(encoding="utf-8").splitlines():
            name, size = line.split("\t")
            sizes[name.removesuffix(".gguf")] = round(int(size) / 1e9, 2)
    rows = []
    for log in sorted(args.results.glob("kld_*_*.log")):
        variant, lang = log.stem.removeprefix("kld_").rsplit("_", 1)
        row = {"varyant": variant, "dil": lang, "boyut_gb": sizes.get(variant)}
        row.update(parse(log.read_text(encoding="utf-8", errors="replace")))
        rows.append(row)
    out = args.out or args.results / "kld.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for r in sorted(rows, key=lambda r: (r["dil"], r["kld_mean"] or 9)):
        print(
            f"{r['dil']} {r['varyant']:14s} {r['boyut_gb']!s:>6s} GB  KLD {r['kld_mean']}  "
            f"p99 {r['kld_p99']}  top1 {r['same_top_p']}%  PPL {r['ppl_q']} (bf16 {r['ppl_base']})"
        )
    print(f"{len(rows)} satır: {out}")
    missing = [r for r in rows if r["kld_mean"] is None]
    return 1 if missing or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
