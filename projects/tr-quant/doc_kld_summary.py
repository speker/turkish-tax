"""doc_kld çıktılarının özeti: aynı belgelerde Türkçe/İngilizce KL oranları, belgeler üzerinde eşli bootstrap.

Kullanım: python doc_kld_summary.py results/<model>/belge [--boot 10000]
Token başına oran = (ΣKL_TR / ΣN_TR) / (ΣKL_EN / ΣN_EN).
İçerik başına oran = ΣKL_TR / ΣKL_EN (iki dilde aynı belgeler, bütün token'lar puanlı). Ayrıca dil içinde varyant kıyası (ör. Türkçe metinde Q4_K_M-tr / Q4_K_M-en).
Sonuç: DİZİN/ozet.json ve ekrana tablo.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path


def load(path: Path) -> list[tuple[int, float, float, float, int]]:
    with open(path, encoding="utf-8") as f:
        return [
            (
                int(r["token"]),
                float(r["kl_toplam"]),
                float(r["nll_taban"]),
                float(r["nll_varyant"]),
                int(r["ayni_top1"]),
            )
            for r in csv.DictReader(f)
        ]


def ci(vals: list[float]) -> tuple[float, float]:
    s = sorted(vals)
    return s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    rng = random.Random(0)
    variants = sorted(p.name[3:-4] for p in a.dir.glob("tr_*.csv") if (a.dir / f"en_{p.name[3:]}").exists())
    data = {v: (load(a.dir / f"tr_{v}.csv"), load(a.dir / f"en_{v}.csv")) for v in variants}
    n_docs = len(next(iter(data.values()))[0])
    idx_sets = [[rng.randrange(n_docs) for _ in range(n_docs)] for _ in range(a.boot)]

    def ratios(tr, en, idx):
        ntr = sum(tr[i][0] for i in idx)
        nen = sum(en[i][0] for i in idx)
        ktr = sum(tr[i][1] for i in idx)
        ken = sum(en[i][1] for i in idx)
        return (ktr / ntr) / (ken / nen), ktr / ken

    out = {}
    for v, (tr, en) in data.items():
        assert len(tr) == len(en) == n_docs
        full = list(range(n_docs))
        tok_r, cont_r = ratios(tr, en, full)
        boot = [ratios(tr, en, idx) for idx in idx_sets]
        row = {
            "token_basina_tr_en": tok_r,
            "token_basina_ci": ci([b[0] for b in boot]),
            "icerik_basina_tr_en": cont_r,
            "icerik_basina_ci": ci([b[1] for b in boot]),
        }
        for lang, rows in (("tr", tr), ("en", en)):
            n = sum(r[0] for r in rows)
            row[f"{lang}_token"] = n
            row[f"{lang}_kl_ort"] = sum(r[1] for r in rows) / n
            row[f"{lang}_ppl_degisim_yuzde"] = (
                math.exp((sum(r[3] for r in rows) - sum(r[2] for r in rows)) / n) - 1
            ) * 100
            row[f"{lang}_ayni_top1_yuzde"] = sum(r[4] for r in rows) / n * 100
        out[v] = row
    pairs = [(f"{t}-tr", f"{t}-en") for t in ("Q4_K_M", "IQ4_XS", "IQ3_M")] + [("Q4_K_M-tr", "Q4_K_M-none")]
    kiyas = {}
    for x, y in pairs:
        if x in data and y in data:
            for li, lang in enumerate(("tr", "en")):
                X, Y = data[x][li], data[y][li]
                full = sum(r[1] for r in X) / sum(r[1] for r in Y)
                boot = [sum(X[i][1] for i in idx) / sum(Y[i][1] for i in idx) for idx in idx_sets]
                kiyas[f"{x} / {y}, {lang} metin"] = {"kl_orani": full, "ci": ci(boot)}
    (a.dir / "ozet.json").write_text(json.dumps({"varyant": out, "kiyas": kiyas}, indent=1), encoding="utf-8")
    print(f"{a.dir}  ({n_docs} belge, {a.boot} eşli bootstrap)")
    for v, r in out.items():
        tl, th = r["token_basina_ci"]
        il, ih = r["icerik_basina_ci"]
        print(
            f"  {v:14s} KL TR {r['tr_kl_ort']:.5f} EN {r['en_kl_ort']:.5f} | token başına TR/EN "
            f"{r['token_basina_tr_en']:.3f} ({tl:.3f}–{th:.3f}) | "
            f"içerik başına {r['icerik_basina_tr_en']:.3f} ({il:.3f}–{ih:.3f})"
        )
    for k, r in kiyas.items():
        print(f"  {k}: {r['kl_orani']:.3f} ({r['ci'][0]:.3f}–{r['ci'][1]:.3f})")


if __name__ == "__main__":
    main()
