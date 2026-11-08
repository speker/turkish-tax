"""GGUF model kartı için tablo: boyut (sizes.tsv) ve FLORES+ KLD (flores/kld.csv); Türkçe imatrix'li
varyantlar.

Kullanım: uv run python projects/tr-quant/gguf_card_table.py MODEL_DİZİNİ
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent / "results"
VARIANTS = ["Q8_0", "Q6_K-tr", "Q5_K_M-tr", "Q4_K_M-tr", "IQ4_XS-tr", "IQ3_M-tr", "Q4_K_M-none"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    name = sys.argv[1]
    sizes = {}
    for line in (ROOT / name / "sizes.tsv").read_text(encoding="utf-8").splitlines():
        f, size = line.split("\t")
        sizes[f.removesuffix(".gguf")] = int(size) / 1e9
    with (ROOT / name / "flores" / "kld.csv").open(encoding="utf-8") as fh:
        kld = {(r["varyant"], r["dil"]): r for r in csv.DictReader(fh)}
    print("| File | Size (GB) | KLD vs BF16, Turkish | KLD vs BF16, English | Same top token, Turkish |")
    print("|---|---|---|---|---|")
    for v in VARIANTS:
        tr, en = kld[(v, "tr")], kld[(v, "en")]
        label = (
            f"{v}.gguf" if v != "Q4_K_M-none" else "Q4_K_M, no importance matrix (reference, not uploaded)"
        )
        print(
            f"| {label} | {sizes[v]:.2f} | {float(tr['kld_mean']):.4f} | {float(en['kld_mean']):.4f} | "
            f"{float(tr['same_top_p']):.1f}% |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
