"""tr-serve ölçümlerini özetler: koşu farkı, hız, 1 milyon kelime başına GPU-saat, TR/EN maliyet oranı.

Kullanım: python summarize.py SONUÇ_DIZINI TOKEN_RATIO_CSV [--out ozet.csv]
SONUÇ_DIZINI altında her model bir klasör: ortam.json ve <dil>_c<eşzamanlılık>_r<koşu>.json.
gpu_saat_1m_kelime = 1e6 * (token/kelime) / (çıktı token/s / TP) / 3600
tr_en_maliyet = (TR/EN token oranı) * (EN çıktı token/s / TR çıktı token/s); aynı içerik için.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

from runs import SPREAD_LIMIT_PCT, tokenizer_repo

# Repo zaten nicelenmiş ağırlıkla geliyor (model_dtype.py); "-fp8" eki olmayan koşu bu biçimdedir.
NATIVE = {
    "ministral-3-14b": "fp8-yerel",
    "gpt-oss-120b": "mxfp4-yerel",
    "gpt-oss-120b-sicak-a": "mxfp4-yerel",
    "gpt-oss-120b-sicak-b": "mxfp4-yerel",
}
RUN = re.compile(r"^(tr|en)_c(\d+)_r(\d+)\.json$")
FIELDS = [
    "model",
    "dtype",
    "tp",
    "dil",
    "eszamanlilik",
    "kosu",
    "cikti_tok_s",
    "kosu_farki_yuzde",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "token_kelime",
    "gpu_saat_1m_kelime",
    "tr_en_maliyet",
]


def ratios(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return {r["model"]: r for r in csv.DictReader(fh)}


def metric(d: dict, *keys):
    for k in keys:
        if d.get(k) is not None:
            return float(d[k])
    return None


def summarize_model(model_dir: Path, tok: dict) -> list[dict]:
    env = json.loads((model_dir / "ortam.json").read_text(encoding="utf-8"))
    base = model_dir.name.removesuffix("-fp8")
    if model_dir.name.endswith("-fp8"):
        dtype = f"{NATIVE[base]}+fp8" if base in NATIVE else "fp8"
    else:
        dtype = NATIVE.get(base, "bf16")
    r = tok.get(tokenizer_repo(model_dir.name), {})
    runs: dict[tuple, list] = {}
    for f in sorted(model_dir.glob("*.json")):
        m = RUN.match(f.name)
        if m:
            runs.setdefault((m[1], int(m[2])), []).append(json.loads(f.read_text(encoding="utf-8")))
    rows = []
    medians = {}
    for (lang, conc), data in sorted(runs.items()):
        tput = [d["output_throughput"] for d in data]
        median = statistics.median(tput)
        medians[(lang, conc)] = median
        spread = (max(tput) - min(tput)) / median * 100 if len(tput) > 1 else None
        # token/kelime, CSV'deki yuvarlanmış sütundan değil ham sayımlardan
        tpw = int(r[f"{lang}_token"]) / int(r[f"{lang}_kelime"]) if r else None
        per_gpu = median / int(env["tp"])
        rows.append(
            {
                "model": base,
                "dtype": dtype,
                "tp": env["tp"],
                "dil": lang,
                "eszamanlilik": conc,
                "kosu": len(tput),
                "cikti_tok_s": round(median, 1),
                "kosu_farki_yuzde": round(spread, 2) if spread is not None else None,
                # gecikme yüzdelikleri de koşular üzerinden medyan
                "ttft_p50_ms": statistics.median(metric(d, "p50_ttft_ms", "median_ttft_ms") for d in data),
                "ttft_p99_ms": statistics.median(metric(d, "p99_ttft_ms") for d in data),
                "tpot_p50_ms": statistics.median(metric(d, "p50_tpot_ms", "median_tpot_ms") for d in data),
                "token_kelime": tpw,
                "gpu_saat_1m_kelime": round(1e6 * tpw / per_gpu / 3600, 3) if tpw else None,
                "tr_en_maliyet": None,
            }
        )
    for x in rows:
        conc = x["eszamanlilik"]
        if x["dil"] == "tr" and ("en", conc) in medians and r:
            ratio = int(r["tr_token"]) / int(r["en_token"])
            cost = ratio * medians[("en", conc)] / medians[("tr", conc)]
            x["tr_en_maliyet"] = round(cost, 4)
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("token_ratio", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    tok = ratios(args.token_ratio)
    rows = []
    for d in sorted(p for p in args.results.iterdir() if (p / "ortam.json").exists()):
        rows += summarize_model(d, tok)
    out = args.out or args.results / "ozet.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for x in rows:
        flag = " !" if x["kosu_farki_yuzde"] and x["kosu_farki_yuzde"] > SPREAD_LIMIT_PCT else ""
        print(
            f"{x['model']:20s} {x['dtype']:4s} {x['dil']} c={x['eszamanlilik']:<3d} "
            f"{x['cikti_tok_s']:>8.1f} tok/s  fark {x['kosu_farki_yuzde']}%{flag}  "
            f"GPU-saat/1M kelime {x['gpu_saat_1m_kelime']}  TR/EN {x['tr_en_maliyet']}"
        )
    print(f"{len(rows)} satır: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
