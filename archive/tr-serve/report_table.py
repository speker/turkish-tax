"""ozet.csv'den rapor tablosu (Markdown): model başına bir satır, verilen eşzamanlılıkta.

Kullanım: python report_table.py OZET_CSV [--c 256]
Ağırlık sütunu repodaki biçimdir (bf16, fp8-yerel, mxfp4-yerel); FP8 hızlanma yalnız BF16 ağırlıklı
modellerde, vLLM'in dinamik FP8'i (--quantization fp8) ile ölçülür.
"""

import argparse
import csv
import sys
from pathlib import Path

BASE_DTYPES = ("bf16", "fp8-yerel", "mxfp4-yerel")


def fmt(x, nd=1):
    if x in (None, ""):
        return "-"
    s = f"{float(x):,.{nd}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("ozet", type=Path)
    parser.add_argument("--c", type=int, default=256)
    args = parser.parse_args()
    with args.ozet.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["eszamanlilik"]) == args.c]
    by = {(r["model"], r["dtype"], r["dil"]): r for r in rows}
    models = sorted({r["model"] for r in rows})
    print(f"Eşzamanlılık {args.c}, tek H200, çıktı 256 token, medyan:\n")
    print(
        "| Model | Ağırlık | TR tok/s | EN tok/s | TR/EN maliyet | GPU-saat / 1M TR kelime | "
        "FP8 hızlanma (TR) | En büyük koşu farkı |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for m in models:
        dtype = next((d for d in BASE_DTYPES if (m, d, "tr") in by), None)
        if dtype is None or (m, dtype, "en") not in by:
            continue
        tr, en = by[(m, dtype, "tr")], by[(m, dtype, "en")]
        tr8 = by.get((m, "fp8", "tr")) if dtype == "bf16" else None
        fp8 = fmt(float(tr8["cikti_tok_s"]) / float(tr["cikti_tok_s"]), 2) + "×" if tr8 else "-"
        spread = max(
            float(r["kosu_farki_yuzde"] or 0) for k, r in by.items() if k[0] == m and k[1] in (dtype, "fp8")
        )
        print(
            f"| {m} | {dtype} | {fmt(tr['cikti_tok_s'])} | {fmt(en['cikti_tok_s'])} | "
            f"{fmt(tr['tr_en_maliyet'], 2)} | {fmt(tr['gpu_saat_1m_kelime'], 3)} | {fp8} | %{fmt(spread)} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
