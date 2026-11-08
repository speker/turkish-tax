"""GGUF KLD sonuçlarının bağımsız denetimi (ham llama-perplexity log'larından).

Kullanım: uv run python projects/tr-quant/verify_kld.py
Her (model, değerlendirme seti, dil) grubunda bütün varyantlar aynı parça sayısı ve aynı BF16 taban PPL'siyle
ölçülmüş mü, log tamamlanmış mı (Mean KLD satırı var mı) ve kld.csv log'larla aynı mı, denetler.
Çıkış 0: bütün denetimler geçti.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent / "results"
CHUNKS = re.compile(r"computing over (\d+) chunks")
BASE = re.compile(r"Mean PPL\(base\)\s*:\s*([\d.]+)")
KLD = re.compile(r"Mean\s+KLD:\s*([\d.]+)")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    errors = []
    groups = defaultdict(list)
    for log in sorted(ROOT.glob("*/**/kld_*.log")):
        variant, lang = log.stem.removeprefix("kld_").rsplit("_", 1)
        text = log.read_text(encoding="utf-8", errors="replace")
        chunks, base, kld = CHUNKS.search(text), BASE.search(text), KLD.search(text)
        if not (chunks and base and kld):
            found = f"parça {bool(chunks)}, taban {bool(base)}, KLD {bool(kld)}"
            errors.append(f"{log.relative_to(ROOT)}: eksik log ({found})")
            continue
        groups[(log.parent.relative_to(ROOT), lang)].append(
            (variant, int(chunks[1]), float(base[1]), float(kld[1]))
        )
    for (where, lang), rows in sorted(groups.items()):
        n = {r[1] for r in rows}
        b = {r[2] for r in rows}
        if len(n) != 1 or len(b) != 1:
            errors.append(f"{where} {lang}: parça {sorted(n)}, taban PPL {sorted(b)}")
        print(f"{str(where):42s} {lang}: {len(rows):2d} varyant, {n.pop()} parça, taban PPL {b.pop()}")
        csv_path = ROOT / where / "kld.csv"
        if csv_path.exists():
            with csv_path.open(encoding="utf-8") as fh:
                rows_csv = {(r["varyant"], r["dil"]): float(r["kld_mean"]) for r in csv.DictReader(fh)}
            for variant, _, _, kld in rows:
                if abs(rows_csv.get((variant, lang), -1) - kld) > 1e-9:
                    errors.append(
                        f"{where} {variant} {lang}: kld.csv {rows_csv.get((variant, lang))} != log {kld}"
                    )
    for e in errors:
        print("HATA ", e)
    print(f"grup {len(groups)}, hata {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
