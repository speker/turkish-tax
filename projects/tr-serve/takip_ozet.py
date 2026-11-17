# ruff: noqa: E501
"""17 Kasım takip ölçümlerinin (h200) özeti: TP ara noktaları ve komşusuz kontrol, uzun girdi (Qwen, Gemma 4,
eşit token), açık döngü, Ministral tek başına, istemci CPU kontrolü, doğal bitiş, belge düzeyinde FP8, Belebele.

Kullanım (h200): python3 takip_ozet.py [--kok ~/work/tr-serve] > ozet.md
Maliyet tanımları makaleyle aynı: TR/EN maliyet = ρ × (EN çıktı tok/s ÷ TR çıktı tok/s); GPU-saat / 1M Türkçe kelime
= 10⁶ × (TR token/kelime) × GPU / (çıktı tok/s) / 3600. Tekrar başına (eşli) değerlerin medyanı ve aralığı.
"""

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

AP = argparse.ArgumentParser()
AP.add_argument("--kok", type=Path, default=Path.home() / "work" / "tr-serve")
AP.add_argument("--token-orani", type=Path, default=Path(__file__).resolve().parent / "token_ratio.csv")
A = AP.parse_args()
R = A.kok
with open(A.token_orani, encoding="utf-8") as f:
    TOK = {r["model"]: r for r in csv.DictReader(f)}
REPO = {
    "qwen": "Qwen/Qwen3.8-27B",
    "gemma-4": "google/gemma-4-12B-it",
    "ministral": "mistralai/Ministral-3-14B-Instruct-2512",
    "turkish-gemma": "ytu-ce-cosmos/Turkish-Gemma-9b-T1",
}
PAT = re.compile(r"^([a-z]+(?:-o\d+)?(?:pad-[a-z]+)?)_c(\d+)_r(\d+)\.json$")


def repo(name: str) -> str:
    return next(v for k, v in REPO.items() if name.startswith(k))


def rho(name: str) -> float:
    t = TOK[repo(name)]
    return int(t["tr_token"]) / int(t["en_token"])


def tpw(name: str) -> float:
    t = TOK[repo(name)]
    return int(t["tr_token"]) / int(t["tr_kelime"])


def load(d: Path) -> dict:
    out = {}
    for f in d.glob("*.json"):
        m = PAT.match(f.name)
        if m:
            out[(m[1], int(m[2]), int(m[3]))] = json.loads(f.read_text())
    return out


def f3(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def med_rng(xs: list[float]) -> str:
    return f"{f3(st.median(xs))} ({f3(min(xs))}–{f3(max(xs))})" if xs else "—"


print("# Takip ölçümleri özeti (h200, 17 Kasım)\n")

print("## 1. TP ara noktaları (Qwen3.8-27B, Türkçe), paralel hatlar ve komşusuz kontrol\n")
print("| GPU | İstek | TPOT ms | Çıktı tok/s | GPU-saat/1M TR kelime | TR/EN maliyet | Komşusuz GPU-saat |")
print("|---|---|---|---|---|---|---|")
for tp in (1, 2, 4):
    run = load(R / "results-takip" / f"qwen3.8-27b-tp{tp}")
    tek = load(R / "results-takip-tek" / f"qwen3.8-27b-tp{tp}")
    for c in sorted({k[1] for k in run}):
        tr = [d for (lg, cc, _), d in run.items() if lg == "tr" and cc == c]
        reps = sorted({r for (lg, cc, r) in run if cc == c})
        costs = [
            rho("qwen") * run[("en", c, r)]["output_throughput"] / run[("tr", c, r)]["output_throughput"]
            for r in reps
            if ("en", c, r) in run and ("tr", c, r) in run
        ]
        gph = [1e6 * tpw("qwen") * tp / d["output_throughput"] / 3600 for d in tr]
        tg = [
            1e6 * tpw("qwen") * tp / d["output_throughput"] / 3600
            for (lg, cc, _), d in tek.items()
            if lg == "tr" and cc == c
        ]
        print(
            f"| {tp} | {c} | {st.median(d['median_tpot_ms'] for d in tr):.1f} | "
            f"{st.median(d['output_throughput'] for d in tr):.0f} | {med_rng(gph)} | {med_rng(costs)} | "
            f"{med_rng(tg) if tg else '—'} |"
        )

print("\n## 2. Uzun girdi\n")
for name in ("qwen3.8-27b-uzun", "gemma-4-12b-it-uzun", "qwen3.8-27b-uzun-esit"):
    run = load(R / "results-takip" / name)
    if not run:
        continue
    langs = sorted({k[0] for k in run})
    print(f"### {name} ({', '.join(langs)})\n")
    print(
        "| İstek | " + " | ".join(f"{lg} tok/s" for lg in langs) + " | girdi TR/EN | EN/TR hız (256) | "
        "içerik eşleşmeli maliyet |"
    )
    print("|---|" + "---|" * (len(langs) + 3))
    for c in sorted({k[1] for k in run}):

        def med(lg: str, key: str = "output_throughput", cc: int = c, rr: dict = run) -> float | None:
            xs = [d[key] for (l2, c2, _), d in rr.items() if l2 == lg and c2 == cc]
            return st.median(xs) if xs else None

        def inp(lg: str, cc: int = c, rr: dict = run) -> float | None:
            xs = [
                d["total_input_tokens"] / d["completed"]
                for (l2, c2, _), d in rr.items()
                if l2 == lg and c2 == cc
            ]
            return st.median(xs) if xs else None

        en, tr256 = med("enlong"), med("trlong")
        ratio_in = inp("trlong") / inp("enlong") if inp("trlong") and inp("enlong") else None
        # İçerik eşleşmeli: Türkçe çıktı 256 × girdi token oranı; maliyet = (TR GPU-süresi / istek) / (EN GPU-süresi / istek)
        cm = next((lg for lg in langs if lg.startswith("trlong-o")), None)
        cm_cost = None
        if cm and en and med(cm):
            n_tr = int(cm.split("-o")[1])
            cm_cost = (n_tr / med(cm)) / (256 / en)
        cells = " | ".join(f"{med(lg):.0f}" if med(lg) else "—" for lg in langs)
        print(
            f"| {c} | {cells} | {f3(ratio_in) if ratio_in else '—'} | {f3(en / tr256) if en and tr256 else '—'} | "
            f"{f3(cm_cost) if cm_cost else '—'} |"
        )
    print()

print("## 3. Açık döngü (Qwen3.8-27B, 1 GPU, Poisson, 256 çıktı token)\n")
print("| İstek/sn | Dil | Karşılanan istek/sn | TTFT p50 / p99 ms | TPOT p50 / p99 ms |")
print("|---|---|---|---|---|")
for rate in (4, 8, 12):
    run = load(R / "results-takip" / f"qwen3.8-27b-acik-r{rate}")
    for lg in ("tr", "en"):
        xs = [d for (l2, _, _), d in run.items() if l2 == lg]
        if xs:
            g = lambda k, xs=xs: st.median(d[k] for d in xs)  # noqa: E731
            print(
                f"| {rate} | {lg} | {g('request_throughput'):.2f} | {g('median_ttft_ms'):.0f} / {g('p99_ttft_ms'):.0f} | "
                f"{g('median_tpot_ms'):.1f} / {g('p99_tpot_ms'):.1f} |"
            )

print("\n## 4. Ministral 3 14B, 256 istek, h200'de tek başına (5 tekrar)\n")
run = load(R / "results-takip-tek" / "ministral-3-14b-bf16")
reps = sorted({r for (_, _, r) in run})
costs = [
    rho("ministral") * run[("en", 256, r)]["output_throughput"] / run[("tr", 256, r)]["output_throughput"]
    for r in reps
    if ("en", 256, r) in run and ("tr", 256, r) in run
]
print(
    f"Tekrar başına TR/EN maliyet: {', '.join(f3(c) for c in costs)}; medyan {med_rng(costs)}. "
    "Makale (2 koşu): 1,575 ve 1,665 (yayın 1,619); yedek makine 5 tekrar: 1,647–1,663.\n"
)

print("## 5. İstemci CPU kontrolü (Turkish-Gemma-9b-T1, 1 GPU)\n")
print("| İstek | Dil | İstemci sunucuyla aynı CPU | İstemci ayrı CPU | Fark |")
print("|---|---|---|---|---|")
a, b = (
    load(R / "results-takip-istemci" / "turkish-gemma-9b-t1"),
    load(R / "results-takip-istemci" / "turkish-gemma-9b-t1-ayri"),
)
for c in (64, 256):
    for lg in ("tr", "en"):
        x = [d["output_throughput"] for (l2, c2, _), d in a.items() if l2 == lg and c2 == c]
        y = [d["output_throughput"] for (l2, c2, _), d in b.items() if l2 == lg and c2 == c]
        if x and y:
            print(
                f"| {c} | {lg} | {st.median(x):.0f} | {st.median(y):.0f} | {(st.median(y) / st.median(x) - 1) * 100:+.2f}% |"
            )

print("\n## 6. Doğal bitiş (düşünme kapalı)\n")
print("| Model | Dil | Tohum | Sınıra dayanan | Token/kelime | FLORES+ | Fark |")
print("|---|---|---|---|---|---|---|")
for d in sorted((R / "results-dogal").glob("*/ozet.json")):
    for k, v in json.loads(d.read_text()).items():
        lg, s = k.split("_")
        print(
            f"| {d.parent.name} | {lg} | {s} | {v['sinira_dayanan']}/{v['istek']} | {v['token_kelime']} | "
            f"{v['flores_token_kelime']} | {v['fark_yuzde']:+}% |"
        )

print("\n## 7. Belge düzeyinde FP8 kalitesi (562 belge, eşli bootstrap)\n")
print("| Model | Kıyas | TR PPL oranı (GA) | EN PPL oranı (GA) | İçerik başına ek kayıp TR/EN (GA) |")
print("|---|---|---|---|---|")
for d in sorted((R / "results" / "fp8-kalite-belge").glob("*")):
    for other in ("fp8", "fp8-yerel", "bf16b"):
        oz, es = d / f"ozet_{other}.json", d / f"eslesik_{other}.json"
        if not oz.exists():
            continue
        o = json.loads(oz.read_text())
        e = json.loads(es.read_text()) if es.exists() else {}

        def ppl(lg: str, o: dict = o) -> str:
            ci = o[lg].get("ppl_oran_ci95", [0, 0])
            return f"{o[lg]['ppl_oran']:.4f} ({ci[0]:.4f}–{ci[1]:.4f})"

        ic = e.get("icerik_basina_ek_kayip_tr_en")
        ici = e.get("icerik_ci95", [None, None])
        ic_s = f"{ic:.3f} ({ici[0]:.3f}–{ici[1]:.3f})" if isinstance(ic, float) and ic == ic else "tanımsız"
        print(f"| {d.name} | bf16 → {other} | {ppl('tr')} | {ppl('en')} | {ic_s} |")

print("\n## 8. Belebele (900 paralel soru), BF16 → FP8\n")
print("| Model | Yöntem | BF16 TR / EN | FP8 TR / EN | Fark TR (GA) | Fark EN (GA) | TR − EN (GA) |")
print("|---|---|---|---|---|---|---|")
for d in sorted((R / "results" / "belebele-fp8").glob("*")):
    for tag, mode in (("fp8", "düz metin"), ("fp8-chat", "sohbet")):
        p = d / f"belebele_{tag}.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        ok = min(j["tr"]["dogruluk_base"], j["en"]["dogruluk_base"]) >= 0.5
        note = "" if ok else " (BF16 < 0,5: geçersiz)"
        print(
            f"| {d.name} | {mode}{note} | {j['tr']['dogruluk_base']:.3f} / {j['en']['dogruluk_base']:.3f} | "
            f"{j['tr']['dogruluk_other']:.3f} / {j['en']['dogruluk_other']:.3f} | "
            f"{j['tr']['fark_puan']:+.2f} ({j['tr_fark_ci95'][0]:+.2f}, {j['tr_fark_ci95'][1]:+.2f}) | "
            f"{j['en']['fark_puan']:+.2f} ({j['en_fark_ci95'][0]:+.2f}, {j['en_fark_ci95'][1]:+.2f}) | "
            f"{j['tr_eksi_en_puan']:+.2f} ({j['tr_eksi_en_ci95'][0]:+.2f}, {j['tr_eksi_en_ci95'][1]:+.2f}) |"
        )
