"""Ölçekli çıktı ölçümünün özeti (night_scaled_v3.sh): TR 256, TR ölçekli ve EN 256 aynı sunucuda.

Kullanım: python scaled_summary.py [SONUÇ_DIZINI ...]   (varsayılan results/olcekli, results/olcekli-2)
TR/EN maliyet = token oranı × EN tok/s ÷ TR tok/s; ölçeklide TR tok/s, aynı içerik uzunluğundaki
Türkçe çıktıdan. Çapraz kontrol: ölçekli TR ve EN koşularının süre oranı × (256 × token oranı ÷ ölçekli
uzunluk). Birden çok dizin verilirse koşular birleştirilir; her dizinin kendi sonucu da (sıra etkisi) yazılır.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOTS = [Path(p) for p in sys.argv[1:]] or [HERE / "results" / "olcekli", HERE / "results" / "olcekli-2"]
MODELS = {
    "turkish-gemma-9b-t1": "ytu-ce-cosmos/Turkish-Gemma-9b-T1",
    "gemma-4-12b-it": "google/gemma-4-12B-it",
    "ministral-3-14b-bf16": "mistralai/Ministral-3-14B-Instruct-2512",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load(dirs: list[Path], label: str, conc: int) -> list[dict]:
    files = [f for d in dirs for f in sorted(d.glob(f"{label}_c{conc}_r*.json"))]
    js = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    for j in js:
        assert j["failed"] == 0 and j["completed"] == 16 * conc, (label, conc)
    return js


def med(js: list[dict], key: str) -> float:
    return statistics.median(j[key] for j in js)


def spread(js: list[dict]) -> float:
    x = [j["output_throughput"] for j in js]
    return (max(x) - min(x)) / statistics.median(x) * 100


def factors(dirs: list[Path], label: str, conc: int) -> tuple | None:
    tr, trs, en = load(dirs, "tr", conc), load(dirs, label, conc), load(dirs, "en", conc)
    if not (tr and trs and en):
        return None
    eng = med(en, "output_throughput")
    return tr, trs, en, eng / med(tr, "output_throughput"), eng / med(trs, "output_throughput")


def main() -> int:
    ratios = {
        r["model"]: int(r["tr_token"]) / int(r["en_token"])
        for r in read_csv(HERE / "results" / "token_ratio.csv")
    }
    v3 = {
        (r["model"], int(r["eszamanlilik"])): float(r["tr_en_maliyet"])
        for r in read_csv(HERE / "results" / "serve-v3" / "ozet.csv")
        if r["dtype"] == "bf16" and r["dil"] == "tr"
    }
    print(
        "eşz. | model | token oranı | TR çıktı | koşu (TR/TR ölç./EN) | EN/TR tok/s eşit -> ölçekli | "
        "maliyet v3 | maliyet eşit -> ölçekli | süre çapraz | koşu farkı % (TR, TR ölç., EN)"
    )
    for conc in (64, 256):
        for name, repo in MODELS.items():
            dirs = [r / name for r in ROOTS if (r / name).is_dir()]
            ratio = ratios[repo]
            label = next(p.name.split("_")[0] for d in dirs for p in d.glob("tr-o*_c256_r*.json"))
            length = int(label.removeprefix("tr-o"))
            res = factors(dirs, label, conc)
            if res is None:
                continue
            tr, trs, en, f_eq, f_sc = res
            dur = med(trs, "duration") / med(en, "duration") * (256 * ratio / length)
            print(
                f"{conc} | {name} | {ratio:.3f} | {length} | {len(tr)}/{len(trs)}/{len(en)} | "
                f"{f_eq:.3f} -> {f_sc:.3f} | {v3[(name, conc)]:.3f} | "
                f"{ratio * f_eq:.3f} -> {ratio * f_sc:.3f} | "
                f"{dur:.3f} | {spread(tr):.2f}, {spread(trs):.2f}, {spread(en):.2f}"
            )
            if len(dirs) > 1:
                for d in dirs:
                    one = factors([d], label, conc)
                    if one:
                        eq, sc = ratio * one[3], ratio * one[4]
                        print(f"      {d.parent.name}: maliyet eşit {eq:.3f} -> ölçekli {sc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
