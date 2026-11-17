# ruff: noqa: E501
"""Türkçe vergisi tekrar ölçümü özeti (yedek 5 tekrar + kontroller, h200 TP=2 ara noktalar); makale v3 ile kıyas.

Kullanım (h200): python3 tekrar_ozet.py [--kok ~/work/tr-serve] [--token-orani token_ratio.csv]
"""

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

AP = argparse.ArgumentParser()
AP.add_argument("--kok", type=Path, default=Path.home() / "work" / "tr-serve")
AP.add_argument(
    "--token-orani", type=Path, default=Path(__file__).resolve().parent / "results" / "token_ratio.csv"
)
ARGS = AP.parse_args()
R = ARGS.kok
Y, V3, T2 = R / "results-yedek", R / "results-v3", R / "results-tp2ara" / "qwen3.8-27b-tp2-ara"
with open(ARGS.token_orani, encoding="utf-8") as f:
    TOK = {r["model"]: r for r in csv.DictReader(f)}
REPO = {
    "turkish-gemma-9b-t1": "ytu-ce-cosmos/Turkish-Gemma-9b-T1",
    "gemma-4-12b-it": "google/gemma-4-12B-it",
    "ministral-3-14b-bf16": "mistralai/Ministral-3-14B-Instruct-2512",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "kumru-2b": "vngrs/Kumru-2B",
}
PAT = re.compile(r"^([a-z]+(?:-o\d+)?)_c(\d+)_r(\d+)\.json$")


def repo(name):
    return next(REPO[p] for p in sorted(REPO, key=len, reverse=True) if name.startswith(p))


def load(d):
    out = {}
    for f in d.glob("*.json"):
        m = PAT.match(f.name)
        if m:
            out[(m[1], int(m[2]), int(m[3]))] = json.loads(f.read_text())
    return out


def ratio(name):
    t = TOK[repo(name)]
    return int(t["tr_token"]) / int(t["en_token"]), int(t["tr_token"]) / int(t["tr_kelime"])


def costs(runs, name, conc, reps=None):
    rho = ratio(name)[0]
    per = []
    for (lang, c, r), d in runs.items():
        if lang == "tr" and c == conc and (reps is None or r in reps) and ("en", c, r) in runs:
            per.append(rho * runs[("en", c, r)]["output_throughput"] / d["output_throughput"])
    return per


def med_cost(runs, name, conc, reps=None):
    tr = [
        d["output_throughput"]
        for (lg, c, r), d in runs.items()
        if lg == "tr" and c == conc and (reps is None or r in reps)
    ]
    en = [
        d["output_throughput"]
        for (lg, c, r), d in runs.items()
        if lg == "en" and c == conc and (reps is None or r in reps)
    ]
    if not tr or not en:
        return None, None
    spread = (max(tr) - min(tr)) / st.median(tr) * 100 if len(tr) > 1 else 0
    return ratio(name)[0] * st.median(en) / st.median(tr), spread


def pmed(runs, name, conc, reps=None):
    per = costs(runs, name, conc, reps)
    return st.median(per) if per else None


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}".replace(".", ",")


print(
    "== A. Ana tablo: TR/EN maliyet, yedek = tekrar başına (eşli) maliyetlerin medyanı ve aralığı | "
    "makale h200 (2 tekrar, makaledeki tanım: medyan hızların oranı)"
)
for name, v3name in [
    ("turkish-gemma-9b-t1", "turkish-gemma-9b-t1"),
    ("gemma-4-12b-it", "gemma-4-12b-it"),
    ("ministral-3-14b-bf16", "ministral-3-14b-bf16"),
    ("qwen3.8-27b", "qwen3.8-27b"),
    ("kumru-2b-uzun", "kumru-2b-uzun"),
]:
    y, v = load(Y / name), load(V3 / v3name)
    for conc in (64, 256):
        _, sp = med_cost(y, name, conc)
        cy = pmed(y, name, conc)
        cv, _ = med_cost(v, v3name, conc)
        per = costs(y, name, conc)
        rng = f"{f(min(per))}–{f(max(per))}" if per else "—"
        diff = f"{(cy / cv - 1) * 100:+.1f}%".replace(".", ",") if cy and cv else "—"
        print(
            f"{name:22s} c{conc:<4d} yedek {f(cy)} (tekrar {rng}, TR hız farkı %{f(sp, 1)}) | makale {f(cv)} | fark {diff}"
        )

print("\n== B. Dil sırası (c256): TR önce (1,3,5) ile EN önce (2,4) tekrarların maliyet ortalaması")
for name in ["turkish-gemma-9b-t1", "gemma-4-12b-it", "ministral-3-14b-bf16", "qwen3.8-27b", "kumru-2b-uzun"]:
    y = load(Y / name)
    a, b = costs(y, name, 256, {1, 3, 5}), costs(y, name, 256, {2, 4})
    if a and b:
        print(
            f"{name:22s} TR önce {f(st.mean(a))} | EN önce {f(st.mean(b))} | fark {f((st.mean(b) / st.mean(a) - 1) * 100, 2)}%"
        )

print(
    "\n== C. Komşuluk: Turkish-Gemma komşusuz (tek) ile paralel partideki aynı tekrarlar (1,2), çıktı hızı farkı"
)
tek, par = load(Y / "turkish-gemma-9b-t1-tek"), load(Y / "turkish-gemma-9b-t1")
for lang in ("tr", "en"):
    for conc in (1, 16, 64, 256):
        a = [tek[(lang, conc, r)]["output_throughput"] for r in (1, 2) if (lang, conc, r) in tek]
        b = [par[(lang, conc, r)]["output_throughput"] for r in (1, 2) if (lang, conc, r) in par]
        if a and b:
            print(
                f"{lang} c{conc:<4d} tek {st.median(a):8.1f} | paralel {st.median(b):8.1f} | fark {f((st.median(b) / st.median(a) - 1) * 100, 2)}%"
            )

print("\n== D. Sıcaklık 0 (Turkish-Gemma, aynı GPU): varsayılan örnekleme ile karşılaştırma")
t0 = load(Y / "turkish-gemma-9b-t1-t0")
for conc in (64, 256):
    for lang in ("tr", "en"):
        a = [par[(lang, conc, r)]["output_throughput"] for r in range(1, 6) if (lang, conc, r) in par]
        b = [t0[(lang, conc, r)]["output_throughput"] for r in (1, 2) if (lang, conc, r) in t0]
        if a and b:
            print(
                f"{lang} c{conc:<4d} varsayılan {st.median(a):8.1f} | sıcaklık 0 {st.median(b):8.1f} | fark {f((st.median(b) / st.median(a) - 1) * 100, 2)}%"
            )
    print(
        f"   maliyet c{conc}: varsayılan {f(pmed(par, 'turkish-gemma-9b-t1', conc))} | sıcaklık 0 {f(pmed(t0, 'turkish-gemma-9b-t1', conc))}"
    )

print("\n== E. Doğal bitiş (Turkish-Gemma, en fazla 1024 token, c256): istek başına çıktı token")
dg = load(Y / "turkish-gemma-9b-t1-dogal")
out = {}
for lang in ("tr-o1024", "en-o1024"):
    xs = [d["total_output_tokens"] / d["completed"] for (lg, c, r), d in dg.items() if lg == lang]
    out[lang] = st.mean(xs) if xs else None
    print(f"{lang}: {f(out[lang], 1)} token/istek ({len(xs)} koşu)")
if all(out.values()):
    print(
        f"ρ_gen (TR/EN doğal çıktı uzunluğu) = {f(out['tr-o1024'] / out['en-o1024'])}; FLORES ρ (token oranı) = {f(ratio('turkish-gemma-9b-t1')[0])}"
    )

print("\n== F. Açık döngü (Poisson, Turkish-Gemma, doğal bitiş): medyan TTFT / TPOT, TR ve EN")
for rate in (10, 20):
    a = load(Y / f"turkish-gemma-9b-t1-acik-r{rate}")
    for lang in ("tr-o1024", "en-o1024"):
        xs = [d for (lg, c, r), d in a.items() if lg == lang]
        if xs:
            print(
                f"r{rate} {lang}: TTFT {f(st.median(d['median_ttft_ms'] for d in xs), 1)} ms, TPOT {f(st.median(d['median_tpot_ms'] for d in xs), 1)} ms, istek/sn {f(st.median(d['request_throughput'] for d in xs), 2)} ({len(xs)} koşu)"
            )

# Çıktı iki dilde 256 token; uzun Türkçe istemin ek prefill süresi zaten EN/TR çıktı hızı oranında. Aynı içerik
# maliyeti bu oran ile (girdi oranı × bu oran) arasında: ikincisi prefill'i token oranıyla bir kez daha çarpar.
print("\n== G. Uzun girdi (Qwen3.8-27B, ~2.700 token TR istem, 256 çıktı): EN/TR çıktı hızı oranı ve iki sınır")
ug = load(Y / "qwen3.8-27b-uzun-girdi")
for conc in (1, 16, 64, 128):
    tr = [d for (lg, c, r), d in ug.items() if lg == "trlong" and c == conc]
    en = [d for (lg, c, r), d in ug.items() if lg == "enlong" and c == conc]
    if tr and en:
        ti = st.mean(d["total_input_tokens"] / d["completed"] for d in tr)
        ei = st.mean(d["total_input_tokens"] / d["completed"] for d in en)
        tt = st.median(d["output_throughput"] for d in tr)
        et = st.median(d["output_throughput"] for d in en)
        print(
            f"c{conc:<4d} girdi TR {ti:.0f} / EN {ei:.0f} token (oran {f(ti / ei)}) | çıktı hızı TR {tt:.0f} / EN {et:.0f} | TTFT TR {st.median(d['median_ttft_ms'] for d in tr):.0f} / EN {st.median(d['median_ttft_ms'] for d in en):.0f} ms | EN/TR hız {f(et / tt)} | girdi oranı × hız oranı {f(ti / ei * et / tt)}"
        )

print("\n== H. TP=2 ara noktalar (h200, GPU 1-2, Qwen3.8-27B TR): makaledeki 64 ve 256 ile kıyas")
t2, v2 = load(T2), load(V3 / "qwen3.8-27b-tp2")
tpw = ratio("qwen3.8-27b")[1]
for conc in (32, 64, 128, 256, 384):
    xs = [d for (lg, c, r), d in t2.items() if lg == "tr" and c == conc]
    if not xs:
        continue
    tps = st.median(d["output_throughput"] for d in xs)
    tpot = st.median(d["median_tpot_ms"] for d in xs)
    gh = 1e6 * tpw / (tps / 2) / 3600
    ref = [d["output_throughput"] for (lg, c, r), d in v2.items() if lg == "tr" and c == conc]
    refs = f" | makale {st.median(ref):.1f} ({f((tps / st.median(ref) - 1) * 100, 1)}%)" if ref else ""
    print(f"c{conc:<4d} TR {tps:8.1f} tok/s, TPOT {f(tpot, 1)} ms, GPU-sa/1M kelime {f(gh)}{refs}")
