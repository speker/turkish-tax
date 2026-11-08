"""fp8_quality.py özetlerini (results/fp8-kalite/<model>/ozet_<etiket>.json) Markdown tabloya döker.

Kullanım: python report_fp8.py [SONUÇ_DIZINI]
"""

import json
import sys
from pathlib import Path

DEFAULT = Path(__file__).parent / "results" / "fp8-kalite"


def pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",")


def ratio(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    print("| Model | Karşılaştırma | PPL oranı TR (%95 GA) | PPL oranı EN | Top-1 uyum TR | Top-1 uyum EN |")
    print("|---|---|---|---|---|---|")
    for path in sorted(root.glob("*/ozet_*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        tr, en = r["tr"], r["en"]
        lo, hi = tr["ppl_oran_ci95"]
        print(
            f"| {path.parent.name} | {r['base']} → {r['other']} | {ratio(tr['ppl_oran'])} "
            f"({ratio(lo)}–{ratio(hi)}) | {ratio(en['ppl_oran'])} "
            f"| %{pct(tr['top1_uyum'])} | %{pct(en['top1_uyum'])} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
