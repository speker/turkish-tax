"""belebele_gguf çıktılarının özeti: tür başına doğruluk, BF16'ya göre eşli fark ve Türkçe/İngilizce matris kıyası.

python belebele_gguf_ozet.py DIZIN [--boot 10000]
DIZIN: gold_<dil>.json ve <dil>_<tür>.csv dosyaları (belebele_gguf.sh). Çıktı: DIZIN/ozet.json ve ekrana tablo.
Aralıklar sorular üzerinde eşli bootstrap; çoklu karşılaştırma için düzeltilmez.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

LANGS = ("tr", "en")


def correct(path: Path, gold: np.ndarray) -> np.ndarray:
    with path.open(encoding="utf-8") as f:
        scores = np.array([[float(r[f"p{k}"]) for k in range(4)] for r in csv.DictReader(f)])
    return scores.argmax(1) == gold


def ci(boot: np.ndarray) -> list[float]:
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    gold = {lang: np.array(json.loads((args.dir / f"gold_{lang}.json").read_text())) for lang in LANGS}
    variants = sorted({p.stem.split("_", 1)[1] for p in args.dir.glob("tr_*.csv")})
    variants = [v for v in variants if (args.dir / f"en_{v}.csv").exists()]
    c = {v: {lang: correct(args.dir / f"{lang}_{v}.csv", gold[lang]) for lang in LANGS} for v in variants}
    n = len(gold["tr"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, (args.boot, n))

    def diff(a: str, b: str) -> dict:
        res = {}
        for lang in LANGS:
            d = c[b][lang].astype(float) - c[a][lang].astype(float)
            res[lang] = {"puan": float(d.mean() * 100), "ci95": ci(d[idx].mean(1) * 100)}
        dd = (c[b]["tr"].astype(float) - c[a]["tr"]) - (c[b]["en"].astype(float) - c[a]["en"])
        res["tr_eksi_en"] = {"puan": float(dd.mean() * 100), "ci95": ci(dd[idx].mean(1) * 100)}
        return res

    out = {"soru": n, "bootstrap": args.boot, "dogruluk": {}, "bf16_farki": {}, "matris_dili": {}}
    print(f"{args.dir}  ({n} soru, {args.boot} eşli bootstrap)")
    for v in variants:
        out["dogruluk"][v] = {lang: float(c[v][lang].mean()) for lang in LANGS}
        line = f"  {v:14s} TR {c[v]['tr'].mean():.4f} EN {c[v]['en'].mean():.4f}"
        if v != "bf16" and "bf16" in c:
            d = diff("bf16", v)
            out["bf16_farki"][v] = d
            line += (
                f" | BF16'ya göre TR {d['tr']['puan']:+.2f} ({d['tr']['ci95'][0]:+.2f}, {d['tr']['ci95'][1]:+.2f})"
                f" EN {d['en']['puan']:+.2f} ({d['en']['ci95'][0]:+.2f}, {d['en']['ci95'][1]:+.2f})"
            )
        print(line)
    for v in variants:
        if "-tr" in v and v.replace("-tr", "-en") in c:
            other = v.replace("-tr", "-en")
            d = diff(other, v)
            out["matris_dili"][f"{v} - {other}"] = d
            print(
                f"  {v} - {other}: TR metin {d['tr']['puan']:+.2f} ({d['tr']['ci95'][0]:+.2f}, "
                f"{d['tr']['ci95'][1]:+.2f}), EN metin {d['en']['puan']:+.2f} ({d['en']['ci95'][0]:+.2f}, "
                f"{d['en']['ci95'][1]:+.2f})"
            )
    (args.dir / "ozet.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
