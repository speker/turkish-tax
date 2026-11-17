"""Makale taslağındaki her sayıyı kaynak dosyasından yeniden hesaplar ve karşılaştırır.

Kullanım: uv run python projects/tr-serve/verify_article.py [--servis SONUÇ_DİZİNİ]
Her iddia: (açıklama, makaledeki değer, kaynaktan hesaplanan değer, tolerans). Makale metni okunmaz; değerler
burada elle yazılıdır. Servis iddiaları varsayılan olarak results/serve-v3'ten denetlenir.
Çıkış 0: hepsi tutuyor.
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from runs import SPREAD_LIMIT_PCT

REPO = Path(__file__).resolve().parents[2]
TOKENS = REPO / "projects/tr-serve/results/token_ratio.csv"
FP8 = REPO / "projects/tr-serve/results/fp8-kalite"
KUMRU = REPO / "projects/tr-serve/results/kumru_pretok.json"
QUANT = REPO / "projects/tr-quant/results"
TG, MI, QW, KU2 = "turkish-gemma-9b-t1", "ministral-3-14b-bf16", "qwen3.8-27b", "kumru-2b"


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tokens(repo: str, field: str) -> float:
    row = next(r for r in load_csv(TOKENS) if r["model"] == repo)
    # oranlar CSV'de yuvarlanmış; ham sayımlardan tam hâlleri
    exact = {
        "tr_token_kelime": ("tr_token", "tr_kelime"),
        "en_token_kelime": ("en_token", "en_kelime"),
        "tr_en_orani": ("tr_token", "en_token"),
    }
    if field in exact:
        num, den = exact[field]
        return int(row[num]) / int(row[den])
    return float(row[field])


def fp8(model: str, other: str, lang: str | None = None, field: str | None = None):
    data = json.loads((FP8 / model / f"ozet_{other}.json").read_text(encoding="utf-8"))
    return data[lang][field] if lang else data


def kld(path: str, variant: str, lang: str, field: str = "kld_mean") -> float:
    rows = load_csv(QUANT / path / "kld.csv")
    return float(next(r for r in rows if r["varyant"] == variant and r["dil"] == lang)[field])


def seeds(model: str, imx: str, field: str = "kld_mean") -> list[float]:
    first = kld(f"{model}/flores", f"Q4_K_M-{imx}", "tr", field)
    return [first] + [kld(f"{model}/flores-tohum", f"Q4_K_M-{imx}-s{s}", "tr", field) for s in (1, 2, 3)]


def text_claims() -> list[tuple]:
    c = []
    toks = (
        ("google/gemma-4-12B-it", 1.362, 2.10, 1.24),
        ("ytu-ce-cosmos/Turkish-Gemma-9b-T1", 1.396, 2.15, 1.23),
        ("Qwen/Qwen3.8-27B", 1.425, 2.23, 1.25),
        ("openai/gpt-oss-120b", 1.425, 2.18, 1.22),
        ("mistralai/Ministral-3-14B-Instruct-2512", 1.484, 2.35, 1.27),
        ("Trendyol/Trendyol-LLM-8B-T1", 1.617, 2.54, 1.26),
        ("vngrs/Kumru-2B", 0.638, 1.57, 1.97),
    )
    for repo, ratio, tpw_tr, tpw_en in toks:
        c.append((f"{repo} TR/EN token", ratio, tokens(repo, "tr_en_orani"), 0.0005))
        c.append((f"{repo} TR token/kelime", tpw_tr, tokens(repo, "tr_token_kelime"), 0.005))
        c.append((f"{repo} EN token/kelime", tpw_en, tokens(repo, "en_token_kelime"), 0.005))
    others = [r for r, *_ in toks if r != "vngrs/Kumru-2B"]
    c.append(("diğer altı TR/EN en düşük", 1.36, min(tokens(r, "tr_en_orani") for r in others), 0.005))
    c.append(("diğer altı TR/EN en yüksek", 1.62, max(tokens(r, "tr_en_orani") for r in others), 0.005))
    c.append(
        (
            "diğer altı EN token/kelime en düşük",
            1.22,
            min(tokens(r, "en_token_kelime") for r in others),
            0.005,
        )
    )
    c.append(
        (
            "diğer altı EN token/kelime en yüksek",
            1.27,
            max(tokens(r, "en_token_kelime") for r in others),
            0.005,
        )
    )
    c.append(
        (
            "diğer altı TR token/kelime en düşük",
            2.10,
            min(tokens(r, "tr_token_kelime") for r in others),
            0.005,
        )
    )
    c.append(
        (
            "diğer altı TR token/kelime en yüksek",
            2.54,
            max(tokens(r, "tr_token_kelime") for r in others),
            0.005,
        )
    )
    c.append(("FLORES cümle", 2009, tokens("vngrs/Kumru-2B", "cumle"), 0))
    qwen = hf_hub_download(
        "Qwen/Qwen3.8-27B", "tokenizer.json", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    with open(qwen, encoding="utf-8") as fh:
        c.append(("Qwen3.8 sözlük", 248044, len(json.load(fh)["model"]["vocab"]), 0))
    k = json.loads(KUMRU.read_text(encoding="utf-8"))
    tr0 = k["kumru_tr_token"]
    c += [
        ("Kumru çok kelimeli token", 8053, k["cok_kelimeli_token"], 0),
        ("Kumru sözlük", 50176, k["sozluk"], 0),
        ("Kumru harf ve yeni satır içeren token", 1, k["harf_ve_yeni_satir_iceren_token"], 0),
        ("Kumru harf ve tab içeren token", 26, k["harf_ve_tab_iceren_token"], 0),
        ("Kumru çok kelimeli FLORES TR %", 8.5, k["tr_tokenlarin_cok_kelimeli_orani"] * 100, 0.05),
        ("Kumru karşı-olgusal TR artışı en az %", 17, (k["bosluk_arkada_tr_token"] / tr0 - 1) * 100, 0.5),
        ("Kumru karşı-olgusal TR artışı en çok %", 29, (k["bosluk_onde_tr_token"] / tr0 - 1) * 100, 0.5),
        ("Kumru karşı-olgusal TR/EN en düşük", 0.60, k["bosluk_arkada_tr_en"], 0.005),
        ("Kumru karşı-olgusal TR/EN en yüksek", 0.76, k["bosluk_onde_tr_en"], 0.005),
        (
            "Kumru boşluk arkada EN artışı %",
            25,
            (k["bosluk_arkada_en_token"] / k["kumru_en_token"] - 1) * 100,
            0.5,
        ),
        (
            "Kumru boşluk önde EN artışı %",
            8,
            (k["bosluk_onde_en_token"] / k["kumru_en_token"] - 1) * 100,
            0.5,
        ),
    ]
    # Kumru: farkın İngilizce payı, öteki altı tokenizer'ın geometrik ortalamasına göre log ölçekte
    others = [r for r, *_ in toks if r != "vngrs/Kumru-2B"]
    en_gm = statistics.geometric_mean(tokens(r, "en_token_kelime") for r in others)
    tr_gm = statistics.geometric_mean(tokens(r, "tr_token_kelime") for r in others)
    e = math.log(tokens("vngrs/Kumru-2B", "en_token_kelime") / en_gm)
    t = math.log(tr_gm / tokens("vngrs/Kumru-2B", "tr_token_kelime"))
    c.append(("Kumru İngilizce payı %", 56, e / (e + t) * 100, 0.5))
    # FP8 kalite
    table = (
        (TG, "fp8", 0.81, 0.34, (0.57, 1.04), (0.17, 0.51), 94.9, 95.7),
        (QW, "fp8", 0.40, 0.39, (0.31, 0.50), (0.30, 0.49), 95.9, 96.2),
        (MI, "fp8", 0.30, 0.27, (0.22, 0.39), (0.19, 0.35), 95.6, 96.2),
        (MI, "fp8-yerel", 0.15, 0.03, (0.06, 0.24), (-0.05, 0.11), 95.5, 96.1),
        (KU2, "fp8", 0.41, 0.38, (0.26, 0.56), (0.28, 0.48), 93.7, 94.9),
    )
    for m, other, tr, en, tr_ci, en_ci, top_tr, top_en in table:
        c.append((f"{m} {other} PPL TR %", tr, (fp8(m, other, "tr", "ppl_oran") - 1) * 100, 0.005))
        c.append((f"{m} {other} PPL EN %", en, (fp8(m, other, "en", "ppl_oran") - 1) * 100, 0.005))
        for lang, ci in (("tr", tr_ci), ("en", en_ci)):
            lo, hi = fp8(m, other, lang, "ppl_oran_ci95")
            c.append((f"{m} {other} {lang} GA alt", ci[0], (lo - 1) * 100, 0.005))
            c.append((f"{m} {other} {lang} GA üst", ci[1], (hi - 1) * 100, 0.005))
        c.append((f"{m} {other} top1 TR", top_tr, fp8(m, other, "tr", "top1_uyum") * 100, 0.05))
        c.append((f"{m} {other} top1 EN", top_en, fp8(m, other, "en", "top1_uyum") * 100, 0.05))
    dyn = [(fp8(m, "fp8", "tr", "ppl_oran") - 1) * 100 for m in (TG, QW, MI, KU2)]
    c.append(("dinamik FP8 TR PPL artışı en düşük", 0.30, min(dyn), 0.005))
    c.append(("dinamik FP8 TR PPL artışı en yüksek", 0.81, max(dyn), 0.005))
    dyn_en = [(fp8(m, "fp8", "en", "ppl_oran") - 1) * 100 for m in (TG, QW, MI, KU2)]
    c.append(("dinamik FP8 EN PPL artışı en düşük", 0.27, min(dyn_en), 0.005))
    c.append(("dinamik FP8 EN PPL artışı en yüksek", 0.39, max(dyn_en), 0.005))
    c.append(("FP8 parça sayısı", 101, fp8(TG, "fp8", "tr", "parca"), 0))
    for m, other, d, lo, hi in ((TG, "fp8", 0.47, 0.18, 0.75), (MI, "fp8-yerel", 0.12, 0.00, 0.25)):
        r = json.loads((FP8 / m / f"eslesik_{other}.json").read_text(encoding="utf-8"))
        c.append((f"{m} {other} eşleşik fark", d, r["fark_tr_eksi_en"] * 100, 0.005))
        c.append((f"{m} {other} eşleşik GA alt", lo, r["ci95"][0] * 100, 0.005))
        c.append((f"{m} {other} eşleşik GA üst", hi, r["ci95"][1] * 100, 0.005))
    for m in (TG, MI):
        null = fp8(m, "bf16b")
        for lang in ("tr", "en"):
            exact = null[lang]["ppl_oran"] == 1.0 and null[lang]["ort_mutlak_logprob_farki"] == 0.0
            c.append(
                (
                    f"{m} BF16 ikinci koşu {lang} birebir aynı",
                    1,
                    float(exact and null[lang]["top1_uyum"] == 1.0),
                    0,
                )
            )
    for m in (QW, MI, KU2):
        a, b = json.loads((FP8 / m / "eslesik_fp8.json").read_text(encoding="utf-8"))["ci95"]
        c.append((f"{m} fp8 eşleşik GA sıfırı içeriyor", 1, float(a < 0 < b), 0))
    for m, v, lo, hi in (
        (TG, 3.3, 1.9, 6.5),
        (MI, 1.7, 1.1, 2.5),
        (QW, 1.4, 1.0, 2.0),
        (KU2, 0.69, 0.41, 1.08),
    ):
        r = json.loads((FP8 / m / "eslesik_fp8.json").read_text(encoding="utf-8"))
        tol = 0.05 if v >= 1 else 0.005
        c.append((f"{m} içerik başına ek kayıp TR/EN", v, r["icerik_basina_ek_kayip_tr_en"], tol))
        c.append((f"{m} içerik GA alt", lo, r["icerik_ci95"][0], tol))
        c.append((f"{m} içerik GA üst", hi, r["icerik_ci95"][1], tol))
    chunk_ratio = [fp8(m, "fp8", "tr", "token") / fp8(m, "fp8", "en", "token") for m in (TG, QW, MI)]
    c.append(("FP8 parçalarında TR/EN token en düşük", 1.4, min(chunk_ratio), 0.05))
    c.append(("FP8 parçalarında TR/EN token en yüksek", 1.5, max(chunk_ratio), 0.05))
    # GGUF
    for path, ratio in (
        ("turkish-gemma-9b-t1/flores", 1.68),
        ("trendyol-llm-8b-t1/flores", 1.58),
        ("kumru-2b/flores", 1.09),
    ):
        r = kld(path, "Q4_K_M-none", "tr") / kld(path, "Q4_K_M-none", "en")
        c.append((f"{path} imatrix'siz TR/EN KLD", ratio, r, 0.005))
    c.append(("TG imatrix'siz KLD", 0.0546, kld("turkish-gemma-9b-t1/flores", "Q4_K_M-none", "tr"), 0.00005))
    c.append(("TG TR imatrix KLD", 0.0358, kld("turkish-gemma-9b-t1/flores", "Q4_K_M-tr", "tr"), 0.00005))
    gain = 1 - kld("turkish-gemma-9b-t1/flores", "Q4_K_M-tr", "tr") / kld(
        "turkish-gemma-9b-t1/flores", "Q4_K_M-none", "tr"
    )
    c.append(("TG imatrix kazancı %", 34, gain * 100, 0.5))
    tg_tr, tg_en = seeds("turkish-gemma-9b-t1", "tr"), seeds("turkish-gemma-9b-t1", "en")
    ty_tr, ty_en = seeds("trendyol-llm-8b-t1", "tr"), seeds("trendyol-llm-8b-t1", "en")
    ty_tr_med, ty_en_med = (
        seeds("trendyol-llm-8b-t1", "tr", "kld_median"),
        seeds("trendyol-llm-8b-t1", "en", "kld_median"),
    )
    c += [
        ("TG TR imatrix en düşük", 0.0355, min(tg_tr), 0.00005),
        ("TG TR imatrix en yüksek", 0.0360, max(tg_tr), 0.00005),
        ("TG EN imatrix en düşük", 0.0365, min(tg_en), 0.00005),
        ("TG EN imatrix en yüksek", 0.0372, max(tg_en), 0.00005),
        ("TG TR < EN dört tohumda", 1, float(max(tg_tr) < min(tg_en)), 0),
        ("TG kazanç %", 3, (1 - sum(tg_tr) / sum(tg_en)) * 100, 0.5),
        ("Trendyol TR imatrix ort.", 0.0217, sum(ty_tr) / 4, 0.00005),
        ("Trendyol EN imatrix ort.", 0.0218, sum(ty_en) / 4, 0.00005),
        ("Trendyol TR imatrix en düşük", 0.0185, min(ty_tr), 0.00005),
        ("Trendyol TR imatrix en yüksek", 0.0271, max(ty_tr), 0.00005),
        ("Trendyol örnek etkisi %", 46, (max(ty_tr) / min(ty_tr) - 1) * 100, 0.5),
        ("Trendyol EN imatrix örnek etkisi %", 21, (max(ty_en) / min(ty_en) - 1) * 100, 0.5),
        ("Trendyol EN imatrix en düşük", 0.0202, min(ty_en), 0.00005),
        ("Trendyol EN imatrix en yüksek", 0.0245, max(ty_en), 0.00005),
        ("Trendyol ilk tek örnek %", 24, (1 - ty_tr[0] / ty_en[0]) * 100, 0.5),
        ("Trendyol medyan TR imatrix ~", 0.0105, sum(ty_tr_med) / 4, 0.00005),
        ("Trendyol medyan EN imatrix en düşük", 0.0115, min(ty_en_med), 0.00005),
        ("Trendyol medyan EN imatrix en yüksek", 0.0118, max(ty_en_med), 0.00005),
        ("Trendyol medyan TR < EN dört tohumda", 1, float(max(ty_tr_med) < min(ty_en_med)), 0),
    ]
    med_gain = [(1 - t / e) * 100 for t, e in zip(ty_tr_med, ty_en_med, strict=True)]
    c.append(("Trendyol medyan kazancı tohum başına en az %", 9, min(med_gain), 0.5))
    c.append(("Trendyol medyan kazancı tohum başına en çok %", 11, max(med_gain), 0.5))
    ids = [json.loads((QUANT / "ortak/ids.json").read_text(encoding="utf-8"))] + [
        json.loads((QUANT / f"ortak/ids_seed{s}.json").read_text(encoding="utf-8")) for s in (1, 2, 3)
    ]
    shared = max(
        len(set(a[f"calib_{lang}"]) & set(b[f"calib_{lang}"]))
        for lang in ("tr", "en")
        for i, a in enumerate(ids)
        for b in ids[i + 1 :]
    )
    c.append(("kalibrasyon örnekleri en fazla ortak makale", 2, shared, 0))
    c.append(("kalibrasyon örnekleri aynı kaynak", 1, float(len({x["source"] for x in ids}) == 1), 0))
    log = (QUANT / "kumru-2b/tokenize_check_noescape.log").read_text(encoding="utf-8").splitlines()
    c.append(("Kumru tokenizasyon metin sayısı", 7, sum(line.startswith("AYNI") for line in log), 0))
    c.append(("Kumru tokenizasyon farklı metin", 0, sum(line.startswith("FARK") for line in log), 0))
    total = sum(int(line.rsplit(" ", 2)[-2]) for line in log if line.startswith("AYNI"))
    c.append(("Kumru tokenizasyon toplam token", 807064, total, 0))
    return c


def serving_claims(root: Path) -> list[tuple]:
    with (root / "ozet.csv").open(encoding="utf-8") as fh:
        o = {(r["model"], r["dtype"], r["dil"], int(r["eszamanlilik"])): r for r in csv.DictReader(fh)}

    def v(m, dt, lang, c, f):
        return float(o[(m, dt, lang, c)][f])

    c = []
    rows = (
        (TG, 10207, 11144, 1.52, 0.059, 1.22),
        ("gemma-4-12b-it", 8079, 8625, 1.45, 0.072, 1.18),
        (MI, 10066, 10982, 1.62, 0.065, 1.12),
        (QW, 3971, 4235, 1.52, 0.156, 1.23),
    )
    for m, trt, ent, cost, gh, sp in rows:
        c.append((f"{m} TR tok/s", trt, v(m, "bf16", "tr", 256, "cikti_tok_s"), 0.5))
        c.append((f"{m} EN tok/s", ent, v(m, "bf16", "en", 256, "cikti_tok_s"), 0.5))
        c.append((f"{m} TR/EN", cost, v(m, "bf16", "tr", 256, "tr_en_maliyet"), 0.005))
        c.append((f"{m} GPU-h", gh, v(m, "bf16", "tr", 256, "gpu_saat_1m_kelime"), 0.0005))
        speed = v(m, "fp8", "tr", 256, "cikti_tok_s") / v(m, "bf16", "tr", 256, "cikti_tok_s")
        c.append((f"{m} FP8 hızlanma", sp, speed, 0.005))
    vendor = v("ministral-3-14b", "fp8-yerel", "tr", 256, "cikti_tok_s") / v(
        MI, "bf16", "tr", 256, "cikti_tok_s"
    )
    c.append(("Ministral üretici FP8", 1.11, vendor, 0.005))
    for conc, sp in ((1, 1.54), (16, 1.49), (64, 1.43)):
        speed = v(MI, "fp8", "tr", conc, "cikti_tok_s") / v(MI, "bf16", "tr", conc, "cikti_tok_s")
        c.append((f"Ministral FP8 c={conc}", sp, speed, 0.005))
    # TP: yüksek eşzamanlılık ölçümleri (tp_high_v3.sh, tp1024_v3.sh); her düzenin en iyi noktası
    tp_cells = {
        1: [("qwen3.8-27b-yuksek", 256), ("qwen3.8-27b-yuksek", 512)],
        2: [("qwen3.8-27b-tp2-yuksek", 256), ("qwen3.8-27b-tp2-yuksek", 512), ("qwen3.8-27b-tp2-1024", 1024)],
        4: [("qwen3.8-27b-tp4-yuksek", 256), ("qwen3.8-27b-tp4-yuksek", 512), ("qwen3.8-27b-tp4-1024", 1024)],
    }
    best = {
        tp: max(cells, key=lambda x: v(x[0], "bf16", "tr", x[1], "cikti_tok_s"))
        for tp, cells in tp_cells.items()
    }
    tput = {tp: v(n, "bf16", "tr", conc, "cikti_tok_s") for tp, (n, conc) in best.items()}
    c.append(("TP1 en iyi tok/s ~", 3950, tput[1], 10))
    c.append(("TP2 en iyi tok/s ~", 7170, tput[2], 10))
    c.append(("TP4 en iyi tok/s ~", 11390, tput[4], 10))
    c.append(("TP2 en iyi noktası 512", 512, best[2][1], 0))
    c.append(("TP4 en iyi noktası 512", 512, best[4][1], 0))
    table_gh = v(QW, "bf16", "tr", 256, "gpu_saat_1m_kelime")
    solo_gh = v("qwen3.8-27b-yuksek", "bf16", "tr", 256, "gpu_saat_1m_kelime")
    c.append(("TP1 tek koşu ile tablo farkı < %1", 1, int(abs(solo_gh / table_gh - 1) < 0.01), 0))
    c.append(("TP2 hızlanma", 1.82, tput[2] / tput[1], 0.005))
    c.append(("TP4 hızlanma", 2.89, tput[4] / tput[1], 0.005))
    for tp, gh in ((1, 0.157), (2, 0.173), (4, 0.218)):
        n, conc = best[tp]
        c.append((f"TP{tp} en iyi GPU-h", gh, v(n, "bf16", "tr", conc, "gpu_saat_1m_kelime"), 0.0005))
    tp_costs = [v(n, "bf16", "tr", conc, "tr_en_maliyet") for cells in tp_cells.values() for n, conc in cells]
    tp_costs += [v(n, "bf16", "tr", 256, "tr_en_maliyet") for n in (QW, "qwen3.8-27b-tp2", "qwen3.8-27b-tp4")]
    c.append(("TP TR/EN en düşük (256 ve üstü)", 1.48, min(tp_costs), 0.005))
    c.append(("TP TR/EN en yüksek", 1.52, max(tp_costs), 0.005))
    tp4_spread = max(
        float(o[(n, "bf16", lang, conc)]["kosu_farki_yuzde"])
        for n, conc in tp_cells[4][1:]
        for lang in ("tr", "en")
    )
    c.append(("TP4 512/1024 koşu farkı en fazla % (6.1)", 6.1, tp4_spread, 0.05))
    c.append(("TP4 1024 TR tok/s ~", 11170, v("qwen3.8-27b-tp4-1024", "bf16", "tr", 1024, "cikti_tok_s"), 10))
    # gecikme hedefinde karşılaştırma (tp_latency.py)
    lat = (
        ("1 GPU c=64", QW, 64, 22.9, 0.257),
        ("4 GPU c=256", "qwen3.8-27b-tp4", 256, 22.1, 0.240),
        ("2 GPU c=16", "qwen3.8-27b-tp2", 16, 10.4, 0.867),
        ("4 GPU c=64", "qwen3.8-27b-tp4", 64, 9.1, 0.421),
    )
    for label, n, conc, tpot, gh in lat:
        c.append((f"{label} TPOT ~22/~10 ms", tpot, v(n, "bf16", "tr", conc, "tpot_p50_ms"), 0.05))
        c.append((f"{label} GPU-h", gh, v(n, "bf16", "tr", conc, "gpu_saat_1m_kelime"), 0.0005))
    c.append(("1 GPU c=1 TPOT ~15 ms", 15, v(QW, "bf16", "tr", 1, "tpot_p50_ms"), 0.5))
    for label, n, conc, sec in (
        ("1 GPU c=256", "qwen3.8-27b-yuksek", 256, 1.9),
        ("1 GPU c=512", "qwen3.8-27b-yuksek", 512, 4.6),
        ("2 GPU c=512", "qwen3.8-27b-tp2-yuksek", 512, 1.3),
        ("4 GPU c=512", "qwen3.8-27b-tp4-yuksek", 512, 1.3),
    ):
        c.append((f"{label} TTFT medyan s", sec, v(n, "bf16", "tr", conc, "ttft_p50_ms") / 1000, 0.05))
    repeat = max(
        abs(v(a, "bf16", "tr", 256, "cikti_tok_s") / v(b, "bf16", "tr", 256, "cikti_tok_s") - 1) * 100
        for a, b in (
            (QW, "qwen3.8-27b-yuksek"),
            ("qwen3.8-27b-tp2", "qwen3.8-27b-tp2-yuksek"),
            ("qwen3.8-27b-tp4", "qwen3.8-27b-tp4-yuksek"),
        )
    )
    c.append(("TP 256 tekrar ölçümleri en fazla %", 1.1, repeat, 0.05))

    def kv_level(n: str, conc: int) -> tuple[float, int]:
        """Bir eşzamanlılık kademesinde (koşan + bekleyen ≈ conc) en yüksek KV doluluğu ve koşan istek."""
        text = (root / n / "server.log").read_text(encoding="utf-8", errors="replace")
        pat = r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%"
        stats = [(int(r), int(w), float(k)) for r, w, k in re.findall(pat, text)]
        sel = [s for s in stats if conc * 0.9 <= s[0] + s[1] <= conc]
        return max(s[2] for s in sel), max(s[0] for s in sel)

    c.append(("1 GPU 256'da KV %", 72, kv_level("qwen3.8-27b-yuksek", 256)[0], 0.5))
    c.append(("2 GPU 512'de KV %", 52, kv_level("qwen3.8-27b-tp2-yuksek", 512)[0], 0.5))
    c.append(("1 GPU 512'de KV dolu %", 100, kv_level("qwen3.8-27b-yuksek", 512)[0], 0.15))
    c.append(("1 GPU 512'de en fazla koşan", 357, kv_level("qwen3.8-27b-yuksek", 512)[1], 0))
    c.append(("2 GPU 1024'te KV dolu %", 100, kv_level("qwen3.8-27b-tp2-1024", 1024)[0], 0.15))
    c.append(("2 GPU 1024'te en fazla koşan", 982, kv_level("qwen3.8-27b-tp2-1024", 1024)[1], 0))
    tp4_kv = max(
        kv_level("qwen3.8-27b-tp4-yuksek", 256)[0],
        kv_level("qwen3.8-27b-tp4-yuksek", 512)[0],
        kv_level("qwen3.8-27b-tp4-1024", 1024)[0],
    )
    c.append(("4 GPU KV %46 altında", 1, int(tp4_kv < 46), 0))
    cfg = json.loads(
        Path(
            hf_hub_download(
                "Qwen/Qwen3.8-27B", "config.json", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
            )
        ).read_text()
    )
    layer_types = cfg.get("text_config", cfg)["layer_types"]
    c.append(("Qwen3.8 doğrusal dikkat katmanı", 48, layer_types.count("linear_attention"), 0))
    c.append(("Qwen3.8 tam dikkat katmanı", 16, layer_types.count("full_attention"), 0))
    # istem uzunluğu (sohbet şablonu dahil) ve TR/EN belirsizliği
    lens = {lang: [] for lang in ("tr", "en")}
    half = []
    for m in (TG, "gemma-4-12b-it", MI, QW):
        for lang in ("tr", "en"):
            j = json.loads((root / m / f"{lang}_c256_r1.json").read_text(encoding="utf-8"))
            lens[lang].append(j["total_input_tokens"] / j["completed"])
        half.append(sum(v(m, "bf16", lang, 256, "kosu_farki_yuzde") for lang in ("tr", "en")) / 2)
    c.append(("TR istem en kısa", 157, min(lens["tr"]), 0.5))
    c.append(("TR istem en uzun", 205, max(lens["tr"]), 0.5))
    c.append(("EN istem en kısa", 115, min(lens["en"]), 0.5))
    c.append(("EN istem en uzun", 159, max(lens["en"]), 0.5))
    c.append(("TR/EN belirsizliği ~±3 %", 3, max(half), 0.5))
    ratio = tokens("vngrs/Kumru-2B", "tr_en_orani")
    runs: dict = {}
    for f in (root / "kumru-2b-uzun").glob("*_r*.json"):
        lang, conc, r = f.stem.split("_")
        runs.setdefault((conc, r), {})[lang] = json.loads(f.read_text(encoding="utf-8"))["output_throughput"]
    per_run = [ratio * d["en"] / d["tr"] for d in runs.values() if len(d) == 2]
    c.append(("Kumru koşu başına TR/EN en düşük", 0.53, min(per_run), 0.005))
    c.append(("Kumru koşu başına TR/EN en yüksek", 0.84, max(per_run), 0.005))
    spread = max(float(x["kosu_farki_yuzde"]) for k, x in o.items() if k[0] == "kumru-2b-uzun")
    c.append(("Kumru koşu farkı en fazla %", 37, spread, 0.5))
    ctrl = json.loads((root / "etkilenme_kontrolu.json").read_text(encoding="utf-8"))
    c.append(("etkilenme en fazla %", 2.4, ctrl["en_buyuk"], 0.05))
    c.append(("etkilenme 256 TR %", -2, ctrl["fark_yuzde"]["tr c256"], 0.5))
    c.append(("etkilenme 256 EN %", -2, ctrl["fark_yuzde"]["en c256"], 0.5))

    # ikinci kontrol: başka bir iş çalışırken, farklı GPU ve CPU bloğunda
    def medians(name: str) -> dict:
        runs: dict = {}
        for f in (root / name).glob("*_r*.json"):
            runs.setdefault(f.stem.rsplit("_", 1)[0], []).append(
                json.loads(f.read_text(encoding="utf-8"))["output_throughput"]
            )
        return {k: statistics.median(x) for k, x in runs.items()}

    tek, k2 = medians("turkish-gemma-9b-t1-tek"), medians("turkish-gemma-9b-t1-kontrol2")
    diffs = [abs(k2[k] / tek[k] - 1) * 100 for k in tek if k in k2]
    c.append(("ikinci kontrol en büyük fark %", 2.75, max(diffs), 0.05))
    # gpt-oss: tam ısınmadan sonra da gürültülü, tabloda yok
    names = ["gpt-oss-120b-sicak-a", "gpt-oss-120b-sicak-b"]
    spreads = {}
    for conc in (1, 16, 64, 256):
        for lang in ("tr", "en"):
            vals = [
                json.loads(f.read_text(encoding="utf-8"))["output_throughput"]
                for n in names
                for f in (root / n).glob(f"{lang}_c{conc}_r*.json")
            ]
            spreads[(lang, conc)] = (max(vals) - min(vals)) / statistics.median(vals) * 100
            c.append((f"gpt-oss {lang} c={conc} sıcak tur sayısı", 4, len(vals), 0))
    hi = [s for (lang, conc), s in spreads.items() if conc >= 16]
    c.append(("gpt-oss 16+ sıcak tur farkı en düşük %", 11, min(hi), 0.5))
    c.append(("gpt-oss 16+ sıcak tur farkı en yüksek %", 30, max(hi), 0.5))
    c.append(
        (
            "gpt-oss c=1 sıcak tur farkı en yüksek %",
            1,
            max(s for (_, conc), s in spreads.items() if conc == 1),
            0.05,
        )
    )
    # dört model: aralıklar
    four = [(m, "bf16") for m, *_ in rows]
    costs = [v(m, dt, "tr", 256, "tr_en_maliyet") for m, dt in four]
    c.append(("TR/EN 256 en düşük", 1.45, min(costs), 0.005))
    c.append(("TR/EN 256 en yüksek", 1.62, max(costs), 0.005))
    slower = [1 - v(m, dt, "tr", 256, "cikti_tok_s") / v(m, dt, "en", 256, "cikti_tok_s") for m, dt in four]
    c.append(("Türkçe token hızı düşüklüğü en az %", 6.2, min(slower) * 100, 0.05))
    c.append(("Türkçe token hızı düşüklüğü en çok %", 8.4, max(slower) * 100, 0.05))
    # dahil etme kuralı: koşular her eşzamanlılıkta %5 içinde
    for m, dt in four + [(f"{m}", "fp8") for m, _ in four]:
        worst = max(float(x["kosu_farki_yuzde"]) for k, x in o.items() if k[0] == m and k[1] == dt)
        c.append((f"{m} {dt} koşu farkı < %5", 1, float(worst < SPREAD_LIMIT_PCT), 0))
    solo = [float(x["kosu_farki_yuzde"]) for k, x in o.items() if k[0] == "gpt-oss-120b" and k[3] >= 16]
    c.append(("gpt-oss tek başına 16+ fark en düşük %", 14, min(solo), 0.5))
    c.append(("gpt-oss tek başına 16+ fark en yüksek %", 49, max(solo), 0.5))
    k_runs: dict = {}
    for f in (root / "kumru-2b-uzun").glob("*_r*.json"):
        lang, conc, r = f.stem.split("_")
        k_runs.setdefault((lang, conc), {})[r] = json.loads(f.read_text(encoding="utf-8"))[
            "output_throughput"
        ]
    r1_dev = max(abs(x["r1"] / ((x["r2"] + x["r3"]) / 2) - 1) * 100 for x in k_runs.values())
    r23 = max(abs(x["r2"] / x["r3"] - 1) * 100 for x in k_runs.values())
    faster = sum(x["r1"] > max(x["r2"], x["r3"]) for x in k_runs.values())
    slower_r1 = sum(x["r1"] < min(x["r2"], x["r3"]) for x in k_runs.values())
    c.append(("Kumru 1. koşu sapması en fazla %", 36, r1_dev, 0.5))
    c.append(("Kumru 2-3. koşu farkı en fazla %", 3.4, r23, 0.05))
    c.append(("Kumru 1. koşu hem hızlı hem yavaş hücre var", 1, float(faster > 0 and slower_r1 > 0), 0))
    repos = {
        TG: "ytu-ce-cosmos/Turkish-Gemma-9b-T1",
        "gemma-4-12b-it": "google/gemma-4-12B-it",
        MI: "mistralai/Ministral-3-14B-Instruct-2512",
        QW: "Qwen/Qwen3.8-27B",
    }
    shares = [
        (tokens(repos[m], "tr_en_orani") - 1) / (v(m, dt, "tr", 256, "tr_en_maliyet") - 1) for m, dt in four
    ]
    c.append(("token payı en düşük %", 76, min(shares) * 100, 0.5))
    c.append(("token payı en yüksek %", 82, max(shares) * 100, 0.5))
    allc = [v(m, dt, "tr", conc, "tr_en_maliyet") for m, dt in four for conc in (1, 16, 64, 256)]
    c.append(("TR/EN tüm eşzamanlılık en düşük", 1.37, min(allc), 0.005))
    c.append(("TR/EN tüm eşzamanlılık en yüksek", 1.62, max(allc), 0.005))
    c.append(("gpt-oss token oranı", 1.425, tokens("openai/gpt-oss-120b", "tr_en_orani"), 0.0005))
    return c


def extra_claims(root: Path) -> list[tuple]:
    with (root / "ozet.csv").open(encoding="utf-8") as fh:
        o = list(csv.DictReader(fh))
    table = (TG, "gemma-4-12b-it", MI, QW)

    def tput(name: str, lang: str, conc: int, run: int) -> float:
        return json.loads((root / name / f"{lang}_c{conc}_r{run}.json").read_text(encoding="utf-8"))[
            "output_throughput"
        ]

    c = []
    widest = max(
        (r for r in o if r["model"] in table and r["dtype"] == "bf16" and r["eszamanlilik"] == "256"),
        key=lambda r: float(r["kosu_farki_yuzde"]),
    )
    c.append(("tablo koşu farkı en fazla %", 4.02, float(widest["kosu_farki_yuzde"]), 0.005))
    c.append(("en geniş hücre Ministral EN", 1, int(widest["model"] == MI and widest["dil"] == "en"), 0))
    tr_words = {int(r["tr_kelime"]) for r in load_csv(TOKENS)}
    en_words = {int(r["en_kelime"]) for r in load_csv(TOKENS)}
    c.append(("TR kelime", 34301, tr_words.pop() if len(tr_words) == 1 else -1, 0))
    c.append(("EN kelime", 42855, en_words.pop() if len(en_words) == 1 else -1, 0))
    tg = next(r for r in load_csv(TOKENS) if r["model"] == "ytu-ce-cosmos/Turkish-Gemma-9b-T1")
    ratio = int(tg["tr_token"]) / int(tg["en_token"])
    costs = [
        ratio * tput(d, "en", 256, r) / tput(d, "tr", 256, r)
        for d in (TG, f"{TG}-tek", f"{TG}-kontrol2")
        for r in (1, 2)
    ]
    c.append(("Turkish-Gemma 3 sunucu maliyet en düşük", 1.50, min(costs), 0.005))
    c.append(("Turkish-Gemma 3 sunucu maliyet en yüksek", 1.54, max(costs), 0.005))
    c1 = [
        abs(
            1
            - statistics.median(tput(m, "tr", 1, r) for r in (1, 2))
            / statistics.median(tput(m, "en", 1, r) for r in (1, 2))
        )
        for m in table
    ]
    c.append(("tek istekte TR/EN farkı en fazla %0,5", 1, int(max(c1) <= 0.005), 0))
    for run, lo, hi in ((1, 6, 9), (2, 6, 11)):
        gaps = [(1 - tput(m, "tr", 256, run) / tput(m, "en", 256, run)) * 100 for m in table]
        c.append((f"256 TR geride, tekrar {run}, en az %", lo, min(gaps), 0.5))
        c.append((f"256 TR geride, tekrar {run}, en çok %", hi, max(gaps), 0.5))
    with (root / "ozet.csv").open(encoding="utf-8") as fh:
        oz = {(r["model"], r["dtype"], r["dil"], int(r["eszamanlilik"])): r for r in csv.DictReader(fh)}
    c.append(
        ("4 GPU c=64 TTFT ms", 468, float(oz[("qwen3.8-27b-tp4", "bf16", "tr", 64)]["ttft_p50_ms"]), 0.5)
    )
    c.append(
        ("2 GPU c=16 TTFT ms", 201, float(oz[("qwen3.8-27b-tp2", "bf16", "tr", 16)]["ttft_p50_ms"]), 0.5)
    )
    ratios = [
        tokens(r, "tr_en_orani")
        for r in (
            "google/gemma-4-12B-it",
            "ytu-ce-cosmos/Turkish-Gemma-9b-T1",
            "Qwen/Qwen3.8-27B",
            "mistralai/Ministral-3-14B-Instruct-2512",
        )
    ]
    c.append(("tablo modelleri token oranı en düşük", 1.36, min(ratios), 0.005))
    c.append(("tablo modelleri token oranı en yüksek", 1.48, max(ratios), 0.005))
    kld = {
        (r["varyant"], r["dil"]): float(r["kld_mean"])
        for r in load_csv(QUANT / "trendyol-llm-8b-t1/flores/kld.csv")
    }
    c.append(("IQ3_M TR matrix EN KLD", 0.0864, kld[("IQ3_M-tr", "en")], 0.00005))
    c.append(("IQ3_M EN matrix EN KLD", 0.0606, kld[("IQ3_M-en", "en")], 0.00005))
    c.append(("IQ3_M artış %", 43, (kld[("IQ3_M-tr", "en")] / kld[("IQ3_M-en", "en")] - 1) * 100, 0.5))
    tg_kld = {
        (r["varyant"], r["dil"]): float(r["kld_mean"])
        for r in load_csv(QUANT / "turkish-gemma-9b-t1/flores/kld.csv")
    }
    c.append(("TG IQ3_M TR matrix EN KLD", 0.0781, tg_kld[("IQ3_M-tr", "en")], 0.00005))
    c.append(("TG IQ3_M EN matrix EN KLD", 0.0779, tg_kld[("IQ3_M-en", "en")], 0.00005))
    hub_kld = pq.read_table(REPO / "projects/tr-serve/hub/data/gguf_kld.parquet").to_pylist()
    iq3_seeds = {r["calibration_seed"] for r in hub_kld if r["quant"] == "IQ3_M"}
    c.append(("IQ3_M yalnız tohum 42", 1, float(iq3_seeds == {42}), 0))
    for model, tr_n, en_n in (("turkish-gemma-9b-t1", 148, 107), ("trendyol-llm-8b-t1", 170, 105)):
        for lang, n in (("tr", tr_n), ("en", en_n)):
            log = (QUANT / model / "flores" / f"kld_Q4_K_M-none_{lang}.log").read_text(encoding="utf-8")
            chunks = int(re.search(r"computing over (\d+) chunks", log)[1])
            c.append((f"{model} {lang} KLD pencere", n, chunks, 0))
    for name in ("qwen3.8-27b-tp2-1024", "qwen3.8-27b-tp4-1024"):
        log = (root / name / "server.log").read_text(encoding="utf-8", errors="replace")
        cap = int(re.search(r"max_cudagraph_capture_size'?:? (\d+)", log)[1])
        running = [int(x) for x in re.findall(r"Running: (\d+) reqs", log)]
        c.append((f"{name} CUDA graph tavanı", 512, cap, 0))
        c.append((f"{name} çoğu zaman 512 üstü", 1, int(sum(x > cap for x in running) > len(running) / 2), 0))
    return c


def scaled_claims() -> list[tuple]:
    roots = [REPO / "projects/tr-serve/results/olcekli", REPO / "projects/tr-serve/results/olcekli-2"]
    stated = {
        TG: ("ytu-ce-cosmos/Turkish-Gemma-9b-T1", 357, 1.529, 1.520),
        "gemma-4-12b-it": ("google/gemma-4-12B-it", 349, 1.447, 1.429),
        MI: ("mistralai/Ministral-3-14B-Instruct-2512", 380, 1.626, 1.483),
        QW: ("Qwen/Qwen3.8-27B", 365, 1.517, 1.406),
    }

    def tput(name: str, label: str, conc: int) -> float:
        files = [f for r in roots for f in (r / name).glob(f"{label}_c{conc}_r*.json")]
        return statistics.median(
            json.loads(f.read_text(encoding="utf-8"))["output_throughput"] for f in files
        )

    c = []
    peaks = []
    for name, (repo, length, eq, sc) in stated.items():
        ratio = tokens(repo, "tr_en_orani")
        c.append((f"ölçekli {name} TR çıktı", length, round(256 * ratio), 0))
        label = f"tr-o{length}"
        en256 = tput(name, "en", 256)
        c.append((f"ölçekli {name} 256 eşit", eq, ratio * en256 / tput(name, "tr", 256), 0.0005))
        c.append((f"ölçekli {name} 256 ölçekli", sc, ratio * en256 / tput(name, label, 256), 0.0005))
        # aynı EN hızına bölündüğü için maliyetin düşmesi TR hızının artmasına denk
        c.append(
            (f"ölçekli {name} 64 maliyet düşük", 1, int(tput(name, label, 64) > tput(name, "tr", 64)), 0)
        )
        logs = [(r / name / "server.log").read_text(encoding="utf-8", errors="replace") for r in roots]
        peaks.append(max(float(x) for log in logs for x in re.findall(r"KV cache usage: ([\d.]+)%", log)))
    c.append(("ölçekli KV tepe en düşük %", 22, min(peaks), 0.5))
    c.append(("ölçekli KV tepe en yüksek %", 72, max(peaks), 0.5))
    return c


FU = REPO / "projects/tr-serve/results/takip-2026-11-17"
TEKRAR = REPO / "projects/tr-serve/results/tekrar-2026-11-17"
DOC = QUANT / "belge-kld"


def jtp(path: Path, field: str = "output_throughput") -> float:
    return json.loads(path.read_text(encoding="utf-8"))[field]


def paired(dirs: list[Path], tr: str, en: str, conc: int, scale: float) -> list[float]:
    """Aynı tekrarın Türkçe ve İngilizce koşusundan aynı içerik maliyeti: scale × EN hız ÷ TR hız.

    scale: çıktılar eşit uzunluktaysa TR/EN token oranı, Türkçe çıktı içerikle uzatıldıysa
    çıktı uzunluklarının oranı.
    """
    out = []
    for d in dirs:
        for f in sorted(d.glob(f"{en}_c{conc}_r*.json")):
            t = d / f.name.replace(f"{en}_", f"{tr}_", 1)
            if t.exists():
                out.append(scale * jtp(f) / jtp(t))
    return out


def doc_kl(model: str, lang: str, variant: str, root: Path = DOC) -> float:
    rows = load_csv(root / model / f"{lang}_{variant}.csv")
    return sum(float(r["kl_toplam"]) for r in rows) / sum(int(r["token"]) for r in rows)


def doc_summary(model: str, variant: str) -> dict:
    """doc_kld_summary.py çıktısı (ozet.txt): token ve içerik başına TR/EN oranı ve bootstrap aralıkları."""
    pat = re.compile(
        rf"^\s+{re.escape(variant)}\s+KL TR [\d.]+ EN [\d.]+ \| "
        r"token başına TR/EN ([\d.]+) \(([\d.]+)–([\d.]+)\) \| içerik başına ([\d.]+) \(([\d.]+)–([\d.]+)\)"
    )
    for line in (DOC / model / "ozet.txt").read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            return dict(
                zip(
                    ("tok", "tok_lo", "tok_hi", "con", "con_lo", "con_hi"),
                    map(float, m.groups()),
                    strict=True,
                )
            )
    raise KeyError(f"{model} {variant}")


def kv_max(path: Path, conc: int) -> float:
    """kv.log'da koşan + bekleyen istek ≈ conc olan satırlarda en yüksek KV doluluğu."""
    pat = r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%"
    stats = [(int(r), int(w), float(k)) for r, w, k in re.findall(pat, path.read_text(encoding="utf-8"))]
    return max(k for r, w, k in stats if conc * 0.9 <= r + w <= conc)


def followup_claims(root: Path) -> list[tuple]:
    """16-17 Kasım takip ölçümlerine dayanan iddialar (makalenin 17 Kasım güncellemesi)."""
    c = []
    with (root / "ozet.csv").open(encoding="utf-8") as fh:
        o = {(r["model"], r["dtype"], r["dil"], int(r["eszamanlilik"])): r for r in csv.DictReader(fh)}
    repo = {
        TG: "ytu-ce-cosmos/Turkish-Gemma-9b-T1",
        "gemma-4-12b-it": "google/gemma-4-12B-it",
        MI: "mistralai/Ministral-3-14B-Instruct-2512",
        QW: "Qwen/Qwen3.8-27B",
    }
    rho = {m: tokens(r, "tr_en_orani") for m, r in repo.items()}
    tpw = {m: tokens(r, "tr_token_kelime") for m, r in repo.items()}
    kosu = FU / "kosu"
    # Ministral: tek başına beş koşu (tablo satırı), ana kampanyanın iki koşusu, ikinci makinenin beş koşusu
    solo = kosu / "takip-tek" / MI
    tr_s = statistics.median(jtp(f) for f in solo.glob("tr_c256_r*.json"))
    en_s = statistics.median(jtp(f) for f in solo.glob("en_c256_r*.json"))
    cost_mi = rho[MI] * en_s / tr_s
    c += [
        ("Ministral tek başına TR tok/s", 10181, tr_s, 0.5),
        ("Ministral tek başına EN tok/s", 11334, en_s, 0.5),
        ("Ministral tek başına TR/EN", 1.65, cost_mi, 0.005),
        ("Ministral tek başına GPU-h", 0.064, 1e6 * tpw[MI] / tr_s / 3600, 0.0005),
    ]
    main_mi = paired([root / MI], "tr", "en", 256, rho[MI])
    later = paired([TEKRAR / "kosu/yedek" / MI, solo], "tr", "en", 256, rho[MI])
    c += [
        ("Ministral ana koşu düşük", 1.575, min(main_mi), 0.0005),
        ("Ministral ana koşu yüksek", 1.665, max(main_mi), 0.0005),
        ("Ministral sonraki on koşu sayısı", 10, len(later), 0),
        ("Ministral sonraki on koşu en düşük", 1.647, min(later), 0.0005),
        ("Ministral sonraki on koşu en yüksek", 1.663, max(later), 0.0005),
        ("Ministral on iki koşu medyanı", 1.65, statistics.median(main_mi + later), 0.005),
    ]
    spreads = [
        float(o[(m, "bf16", lang, 256)]["kosu_farki_yuzde"])
        for m in (TG, "gemma-4-12b-it", QW)
        for lang in ("tr", "en")
    ]
    for lang in ("tr", "en"):
        v = [jtp(f) for f in solo.glob(f"{lang}_c256_r*.json")]
        spreads.append((max(v) - min(v)) / statistics.median(v) * 100)
    c.append(("tablo koşu farkı en fazla % (güncel)", 0.98, max(spreads), 0.005))
    costs = {m: float(o[(m, "bf16", "tr", 256)]["tr_en_maliyet"]) for m in (TG, "gemma-4-12b-it", QW)} | {
        MI: cost_mi
    }
    c += [("TR/EN 256 en düşük (güncel)", 1.45, min(costs.values()), 0.005)]
    c += [("TR/EN 256 en yüksek (güncel)", 1.65, max(costs.values()), 0.005)]
    gaps = [
        1 - float(o[(m, "bf16", "tr", 256)]["cikti_tok_s"]) / float(o[(m, "bf16", "en", 256)]["cikti_tok_s"])
        for m in (TG, "gemma-4-12b-it", QW)
    ]
    gaps.append(1 - tr_s / en_s)
    c += [
        ("TR hız farkı en az % (güncel)", 6.2, min(gaps) * 100, 0.05),
        ("TR hız farkı en çok %", 10.2, max(gaps) * 100, 0.05),
    ]
    shares = [(rho[m] - 1) / (costs[m] - 1) * 100 for m in costs]
    c += [
        ("token payı en düşük % (güncel)", 74, min(shares), 0.5),
        ("token payı en yüksek % (güncel)", 82, max(shares), 0.5),
    ]
    allc = [
        float(o[(m, "bf16", "tr", conc)]["tr_en_maliyet"])
        for m in (TG, "gemma-4-12b-it", QW, MI)
        for conc in (1, 16, 64)
    ]
    allc += list(costs.values())
    c += [
        ("TR/EN 1-256 en düşük", 1.37, min(allc), 0.005),
        ("TR/EN 1-256 en yüksek (güncel)", 1.65, max(allc), 0.005),
    ]
    # dolgulu istem (9 Kasım): Türkçe uzunluğa doldurulmuş İngilizce / Türkçe, aynı çıktı uzunluğu
    pad = []
    for d in (REPO / "projects/tr-serve/results/dolgu").iterdir():
        enpad = [f for f in d.glob("enpad-*_c256_r*.json")]
        out = re.search(r"-o(\d+)_", enpad[0].name)[1]
        pad.append(
            statistics.median(jtp(f) for f in enpad)
            / statistics.median(jtp(f) for f in d.glob(f"tr-o{out}_c256_r*.json"))
        )
    c += [("dolgulu/TR en düşük", 0.992, min(pad), 0.0005), ("dolgulu/TR en yüksek", 1.011, max(pad), 0.0005)]
    # istemci CPU
    a, b = kosu / "takip-istemci" / TG, kosu / "takip-istemci" / f"{TG}-ayri"
    diffs = []
    for lang in ("tr", "en"):
        for conc in (64, 256):
            m1 = statistics.median(jtp(f) for f in a.glob(f"{lang}_c{conc}_r*.json"))
            m2 = statistics.median(jtp(f) for f in b.glob(f"{lang}_c{conc}_r*.json"))
            diffs.append(abs(m2 / m1 - 1) * 100)
    c.append(("istemci ayrı CPU en fazla %", 0.43, max(diffs), 0.005))

    # TP noktaları (takip, paralel hatlar): medyan TPOT ve GPU-saat
    def tp_point(tp: int, conc: int) -> tuple[float, float]:
        d = kosu / "takip" / f"qwen3.8-27b-tp{tp}"
        t = statistics.median(jtp(f, "median_tpot_ms") for f in d.glob(f"tr_c{conc}_r*.json"))
        s = statistics.median(jtp(f) for f in d.glob(f"tr_c{conc}_r*.json"))
        return t, 1e6 * tpw[QW] / s / 3600 * tp

    for label, tp, conc, tpot, gh in (
        ("4 GPU 128", 4, 128, 12.3, 0.282),
        ("2 GPU 160", 2, 160, 23.1, 0.209),
        ("1 GPU 64", 1, 64, 23.1, 0.259),
        ("4 GPU 256", 4, 256, 21.6, 0.239),
    ):
        t, g = tp_point(tp, conc)
        c += [(f"takip {label} TPOT", tpot, t, 0.05), (f"takip {label} GPU-h", gh, g, 0.0005)]
    c.append(("takip 2 GPU 96 TPOT (17 ms sınırı)", 17.3, tp_point(2, 96)[0], 0.05))
    c.append(("takip 1 GPU 96 TPOT (28 ms sınırı)", 27.6, tp_point(1, 96)[0], 0.05))
    c.append(("1 GPU 256 TPOT (57 ms)", 57.1, float(o[(QW, "bf16", "tr", 256)]["tpot_p50_ms"]), 0.05))
    d384 = TEKRAR / "kosu/tp2ara/qwen3.8-27b-tp2-ara"
    gh384 = 1e6 * tpw[QW] / statistics.median(jtp(f) for f in d384.glob("tr_c384_r*.json")) / 3600 * 2
    c.append(("1 GPU 128, 2 GPU 384'ün %4 içinde", 1, float((tp_point(1, 128)[1] / gh384 - 1) * 100 < 4), 0))
    tp_costs = []
    for tp, concs in ((1, (64, 96, 128)), (2, (64, 96, 128, 160)), (4, (64, 96, 128, 256))):
        for conc in concs:
            tp_costs.append(
                statistics.median(paired([kosu / "takip" / f"qwen3.8-27b-tp{tp}"], "tr", "en", conc, rho[QW]))
            )
    c += [
        ("takip TP TR/EN medyan en düşük", 1.47, min(tp_costs), 0.005),
        ("takip TP TR/EN medyan en yüksek", 1.51, max(tp_costs), 0.005),
    ]
    # Kumru İngilizce çok kelimeli token
    k = json.loads(KUMRU.read_text(encoding="utf-8"))
    c.append(("Kumru çok kelimeli FLORES EN %", 3.9, k["en_tokenlarin_cok_kelimeli_orani"] * 100, 0.05))
    # FP8 belge düzeyi
    belge = FU / "fp8-kalite-belge"

    def fj(m: str, name: str) -> dict:
        return json.loads((belge / m / name).read_text(encoding="utf-8"))

    tg = fj(TG, "ozet_fp8.json")
    c += [
        ("TG belge FP8 TR PPL %", 0.55, (tg["tr"]["ppl_oran"] - 1) * 100, 0.005),
        ("TG belge FP8 EN PPL %", 0.88, (tg["en"]["ppl_oran"] - 1) * 100, 0.005),
    ]
    for m, v, lo, hi in (
        (TG, 0.87, 0.57, 1.24),
        (QW, 1.56, 1.10, 2.29),
        (MI, 1.64, 1.18, 2.37),
        (KU2, 0.93, 0.67, 1.27),
    ):
        e = fj(m, "eslesik_fp8.json")
        c += [
            (f"{m} belge içerik TR/EN", v, e["icerik_basina_ek_kayip_tr_en"], 0.005),
            (f"{m} belge içerik GA alt", lo, e["icerik_ci95"][0], 0.005),
            (f"{m} belge içerik GA üst", hi, e["icerik_ci95"][1], 0.005),
        ]
    c.append(
        ("Gemma 4 belge BF16 TR PPL", 1792, fj("gemma-4-12b-it", "ozet_fp8.json")["tr"]["ppl_base"], 0.5)
    )
    rep = json.loads((FU / "fp8-kalite-parca-tekrar" / TG / "eslesik_fp8.json").read_text(encoding="utf-8"))
    orig = json.loads((FP8 / TG / "eslesik_fp8.json").read_text(encoding="utf-8"))
    same = all(
        rep[k] == orig[k] for k in ("fark_tr_eksi_en", "ci95", "icerik_basina_ek_kayip_tr_en", "icerik_ci95")
    )
    c.append(("TG parça tekrarı bit düzeyinde aynı", 1, float(same), 0))
    # Belebele, vLLM BF16 ve FP8
    bb = []
    for p in (FU / "belebele-fp8").glob("*/belebele_fp8*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        bb.append(
            (
                p.parent.name,
                p.stem.endswith("-chat"),
                r,
                min(r["tr"]["dogruluk_base"], r["en"]["dogruluk_base"]) >= 0.5,
            )
        )
    valid = [x for x in bb if x[3]]
    c += [
        ("Belebele geçerli model sayısı", 4, len({x[0] for x in valid}), 0),
        (
            "Belebele FP8 en büyük |fark| puan",
            1.1,
            max(abs(x[2][lang]["fark_puan"]) for x in valid for lang in ("tr", "en")),
            0.05,
        ),
        ("Belebele TR-EN GA üst en büyük", 2.4, max(x[2]["tr_eksi_en_ci95"][1] for x in valid), 0.05),
        (
            "Gemma 4 düz metin EN doğruluk",
            0.28,
            next(x[2]["en"]["dogruluk_base"] for x in bb if x[0] == "gemma-4-12b-it" and not x[1]),
            0.005,
        ),
    ]
    # GGUF belge düzeyi
    for model, tok, tlo, thi, con, clo, chi in (
        ("turkish-gemma-9b-t1", 1.49, 1.45, 1.54, 2.08, 2.02, 2.14),
        ("trendyol-llm-8b-t1", 1.39, 1.35, 1.44, 2.26, 2.18, 2.33),
    ):
        s = doc_summary(model, "Q4_K_M-none")
        for name, v in (
            ("tok", tok),
            ("tok_lo", tlo),
            ("tok_hi", thi),
            ("con", con),
            ("con_lo", clo),
            ("con_hi", chi),
        ):
            c.append((f"{model} belge KL {name}", v, s[name], 0.005))
    ku = doc_summary("kumru-2b", "Q4_K_M-none")
    c += [
        ("Kumru belge KL token başına", 1.18, ku["tok"], 0.005),
        ("Kumru belge KL içerik başına", 0.75, ku["con"], 0.005),
    ]
    cpu = QUANT / "belge-kld-cpu"
    back = []
    for f in (cpu / "kumru-2b").glob("*.csv"):
        lang, variant = f.stem.split("_", 1)
        if variant.startswith(("Q4_K_M", "IQ4_XS")):
            back.append(
                abs(doc_kl("kumru-2b", lang, variant) / doc_kl("kumru-2b", lang, variant, cpu) - 1) * 100
            )
    c.append(("CPU/GPU K-quant farkı en fazla %", 8.5, max(back), 0.05))

    def seeds_doc(model: str, imx: str, quant: str = "Q4_K_M", lang: str = "tr") -> list[float]:
        return [doc_kl(model, lang, f"{quant}-{imx}{s}") for s in ("", "-s1", "-s2", "-s3")]

    tg_tr, tg_en = seeds_doc("turkish-gemma-9b-t1", "tr"), seeds_doc("turkish-gemma-9b-t1", "en")
    ty_tr, ty_en = seeds_doc("trendyol-llm-8b-t1", "tr"), seeds_doc("trendyol-llm-8b-t1", "en")
    tg_none = doc_kl("turkish-gemma-9b-t1", "tr", "Q4_K_M-none")
    c += [
        ("TG belge matrissiz KL", 0.0515, tg_none, 0.00005),
        ("TG belge TR matris ort", 0.0343, statistics.mean(tg_tr), 0.00005),
        ("TG belge matris kazancı %", 33, (1 - statistics.mean(tg_tr) / tg_none) * 100, 0.5),
        (
            "Trendyol belge matris kazancı %",
            44,
            (1 - statistics.mean(ty_tr) / doc_kl("trendyol-llm-8b-t1", "tr", "Q4_K_M-none")) * 100,
            0.5,
        ),
        (
            "Kumru belge matris kazancı %",
            19,
            (1 - doc_kl("kumru-2b", "tr", "Q4_K_M-tr") / doc_kl("kumru-2b", "tr", "Q4_K_M-none")) * 100,
            0.5,
        ),
        (
            "TG kalibrasyon dili kazancı %",
            1.8,
            (1 - statistics.mean(tg_tr) / statistics.mean(tg_en)) * 100,
            0.05,
        ),
        ("TG TR matris en düşük", 0.0341, min(tg_tr), 0.00005),
        ("TG TR matris en yüksek", 0.0346, max(tg_tr), 0.00005),
        ("TG EN matris en düşük", 0.0347, min(tg_en), 0.00005),
        ("TG EN matris en yüksek", 0.0351, max(tg_en), 0.00005),
        (
            "Trendyol kalibrasyon dili kazancı %",
            8.5,
            (1 - statistics.mean(ty_tr) / statistics.mean(ty_en)) * 100,
            0.05,
        ),
        ("Trendyol TR matris en düşük", 0.0199, min(ty_tr), 0.00005),
        ("Trendyol TR matris en yüksek", 0.0201, max(ty_tr), 0.00005),
        ("Trendyol EN matris en düşük", 0.0217, min(ty_en), 0.00005),
        ("Trendyol EN matris en yüksek", 0.0220, max(ty_en), 0.00005),
        (
            "TR matris dört örneklemin her birinde düşük",
            1,
            float(all(t < e for t, e in zip(tg_tr + ty_tr, tg_en + ty_en, strict=True))),
            0,
        ),
        (
            "örneklem yayılımı en fazla %",
            1.4,
            max((max(x) / min(x) - 1) * 100 for x in (tg_tr, tg_en, ty_tr, ty_en)),
            0.05,
        ),
    ]
    ty3 = {
        (imx, lang): seeds_doc("trendyol-llm-8b-t1", imx, "IQ3_M", lang)
        for imx in ("tr", "en")
        for lang in ("tr", "en")
    }
    tg3 = {
        (imx, lang): seeds_doc("turkish-gemma-9b-t1", imx, "IQ3_M", lang)
        for imx in ("tr", "en")
        for lang in ("tr", "en")
    }
    per_seed = [(t / e - 1) * 100 for t, e in zip(ty3[("tr", "en")], ty3[("en", "en")], strict=True)]
    c += [
        (
            "Trendyol IQ3_M EN artış ort %",
            38,
            (statistics.mean(ty3[("tr", "en")]) / statistics.mean(ty3[("en", "en")]) - 1) * 100,
            0.5,
        ),
        ("Trendyol IQ3_M EN artış en az %", 33, min(per_seed), 0.5),
        ("Trendyol IQ3_M EN artış en çok %", 43, max(per_seed), 0.5),
        (
            "Trendyol IQ3_M TR artış %",
            5.5,
            (statistics.mean(ty3[("tr", "tr")]) / statistics.mean(ty3[("en", "tr")]) - 1) * 100,
            0.05,
        ),
        (
            "TG IQ3_M TR düşüş %",
            3.9,
            (1 - statistics.mean(tg3[("tr", "tr")]) / statistics.mean(tg3[("en", "tr")])) * 100,
            0.05,
        ),
        (
            "TG IQ3_M EN düşüş %",
            2.6,
            (1 - statistics.mean(tg3[("tr", "en")]) / statistics.mean(tg3[("en", "en")])) * 100,
            0.05,
        ),
    ]
    # Belebele, GGUF
    gg = {
        m: json.loads((QUANT / "belebele-gguf" / m / "ozet.json").read_text(encoding="utf-8"))
        for m in ("turkish-gemma-9b-t1", "trendyol-llm-8b-t1")
    }
    mild = [v for g in gg.values() for k, v in g["bf16_farki"].items() if not k.startswith("IQ3_M")]
    ty = gg["trendyol-llm-8b-t1"]
    c += [
        (
            "GGUF Belebele Q8/Q4/IQ4 en büyük |fark|",
            1.6,
            max(abs(v[lang]["puan"]) for v in mild for lang in ("tr", "en")),
            0.05,
        ),
        (
            "GGUF Belebele Q8/Q4/IQ4 TR-EN aralığı sıfırı dışlayan yok",
            1,
            float(all(v["tr_eksi_en"]["ci95"][0] < 0 < v["tr_eksi_en"]["ci95"][1] for v in mild)),
            0,
        ),
        ("Trendyol Q4_K_M matrissiz TR kayıp", -1.4, ty["bf16_farki"]["Q4_K_M-none"]["tr"]["puan"], 0.05),
        ("Trendyol Q4_K_M matrissiz GA alt", -2.8, ty["bf16_farki"]["Q4_K_M-none"]["tr"]["ci95"][0], 0.05),
        ("Trendyol Q4_K_M matrissiz GA üst", -0.1, ty["bf16_farki"]["Q4_K_M-none"]["tr"]["ci95"][1], 0.05),
        ("Trendyol Q4_K_M TR matris TR", 0.6, ty["bf16_farki"]["Q4_K_M-tr"]["tr"]["puan"], 0.05),
        (
            "Trendyol Q4_K_M TR matris aralığı sıfırı içeriyor",
            1,
            float(ty["bf16_farki"]["Q4_K_M-tr"]["tr"]["ci95"][0] < 0),
            0,
        ),
        (
            "IQ3_M TR düşüş en az",
            2.7,
            min(-g["bf16_farki"][v]["tr"]["puan"] for g in gg.values() for v in ("IQ3_M-tr", "IQ3_M-en")),
            0.05,
        ),
        (
            "IQ3_M TR düşüş en çok",
            3.7,
            max(-g["bf16_farki"][v]["tr"]["puan"] for g in gg.values() for v in ("IQ3_M-tr", "IQ3_M-en")),
            0.05,
        ),
        (
            "Trendyol IQ3_M TR matris EN kaybı",
            -1.6,
            ty["matris_dili"]["IQ3_M-tr - IQ3_M-en"]["en"]["puan"],
            0.05,
        ),
        (
            "Trendyol IQ3_M TR matris EN GA alt",
            -3.0,
            ty["matris_dili"]["IQ3_M-tr - IQ3_M-en"]["en"]["ci95"][0],
            0.05,
        ),
        (
            "Trendyol IQ3_M TR matris EN GA üst",
            -0.1,
            ty["matris_dili"]["IQ3_M-tr - IQ3_M-en"]["en"]["ci95"][1],
            0.05,
        ),
    ]
    # ikinci makinede tekrar: model başına medyan maliyet, ilk kampanyaya göre
    rel = []
    for m in costs:
        for conc in (64, 256):
            r = statistics.median(paired([TEKRAR / "kosu/yedek" / m], "tr", "en", conc, rho[m]))
            rel.append((r / float(o[(m, "bf16", "tr", conc)]["tr_en_maliyet"]) - 1) * 100)
    c += [("tekrar en düşük %", -0.6, min(rel), 0.05), ("tekrar en yüksek %", 1.8, max(rel), 0.05)]

    # uzun istemler: içerik eşleşmeli maliyet ve girdi token oranı
    def long_costs(session: str, name: str, n_tr: int, concs: tuple) -> tuple[list, list]:
        d = kosu / session / name
        cs, ins = [], []
        for conc in concs:
            cs.append(statistics.median(paired([d], f"trlong-o{n_tr}", "enlong", conc, n_tr / 256)))
            tr = [f for f in d.glob(f"trlong-o{n_tr}_c{conc}_r*.json")]
            ins.append(
                statistics.median(
                    jtp(f, "total_input_tokens")
                    / jtp(d / f.name.replace(f"trlong-o{n_tr}", "enlong"), "total_input_tokens")
                    for f in tr
                )
            )
        return cs, ins

    qc, qi = long_costs("takip", "qwen3.8-27b-uzun", 360, (1, 16, 64, 128))
    gc, gi = long_costs("takip", "gemma-4-12b-it-uzun", 350, (1, 16, 64, 128))
    c += [
        ("Qwen uzun içerik en düşük", 1.405, min(qc), 0.0005),
        ("Qwen uzun içerik en yüksek", 1.433, max(qc), 0.0005),
        ("Gemma 4 uzun içerik en düşük", 1.353, min(gc), 0.0005),
        ("Gemma 4 uzun içerik en yüksek", 1.372, max(gc), 0.0005),
        (
            "uzun içerik / girdi oranı en fazla %",
            2.2,
            max((a / b - 1) * 100 for a, b in zip(qc + gc, qi + gi, strict=True)),
            0.05,
        ),
        ("Qwen uzun girdi oranı", 1.40, statistics.median(qi), 0.005),
        ("Gemma 4 uzun girdi oranı", 1.35, statistics.median(gi), 0.005),
    ]
    kc, ki = long_costs("takip-ek2", "qwen3.8-27b-uzun-kv", 360, (256,))
    c += [
        ("Qwen KV dolu içerik maliyet", 1.47, kc[0], 0.005),
        ("Qwen KV dolu oranın üstü %", 4.9, (kc[0] / ki[0] - 1) * 100, 0.05),
    ]
    c.append(
        ("Qwen 256'da KV dolu %", 100, kv_max(FU / "kv/takip-ek2/qwen3.8-27b-uzun-kv/kv.log", 256), 0.15)
    )
    tc, ti = long_costs("takip-ek2", "turkish-gemma-9b-t1-uzun", 355, (1, 16, 64, 128))
    tgkv = FU / "kv/takip-ek2/turkish-gemma-9b-t1-uzun/kv.log"
    c += [
        ("TG uzun 1 istek", 1.40, tc[0], 0.005),
        ("TG uzun 64 istek", 1.59, tc[2], 0.005),
        ("TG uzun 128 istek", 1.71, tc[3], 0.005),
        ("TG uzun 64'te KV %", 58, kv_max(tgkv, 64), 0.6),
        ("TG uzun 128'de KV dolu %", 100, kv_max(tgkv, 128), 0.15),
        ("TG uzun 128 oranın üstü %", 23, (tc[3] / ti[3] - 1) * 100, 0.5),
    ]
    eq = kosu / "takip-ek3" / f"{TG}-uzun-esit"
    eqr = [
        jtp(f) / jtp(eq / f.name.replace("enlongpad-tg-o355", "trlong-o355"))
        for f in eq.glob("enlongpad-tg-o355_c*_r*.json")
    ]
    c += [
        ("TG eşit token en düşük", 0.997, min(eqr), 0.0005),
        ("TG eşit token en yüksek", 1.000, max(eqr), 0.0005),
    ]

    # açık döngü
    def ol(name: str, label: str, field: str) -> float:
        return statistics.median(
            jtp(f, field)
            for f in (kosu / name.split(":")[0] / name.split(":")[1]).glob(f"{label}_c1024_r*.json")
        )

    c += [
        ("açık 4/sn TR TPOT", 21.6, ol("takip:qwen3.8-27b-acik-r4", "tr", "median_tpot_ms"), 0.05),
        ("açık 4/sn EN TPOT", 21.4, ol("takip:qwen3.8-27b-acik-r4", "en", "median_tpot_ms"), 0.05),
        ("açık 8/sn TR TPOT", 30.9, ol("takip:qwen3.8-27b-acik-r8", "tr", "median_tpot_ms"), 0.05),
        ("açık 8/sn EN TPOT", 30.5, ol("takip:qwen3.8-27b-acik-r8", "en", "median_tpot_ms"), 0.05),
    ]
    d8 = kosu / "takip-ek" / "qwen3.8-27b-acik-icerik-r8"
    e2e = [
        jtp(d8 / f.name.replace("en_", "tr-o365_", 1), "median_e2el_ms") / jtp(f, "median_e2el_ms")
        for f in d8.glob("en_c1024_r*.json")
    ]
    d12 = kosu / "takip-ek" / "qwen3.8-27b-acik-icerik-r12"
    served = [jtp(f, "request_throughput") for f in d12.glob("tr-o365_c1024_r*.json")]
    ttft = [jtp(f, "median_ttft_ms") / 1000 for f in d12.glob("tr-o365_c1024_r*.json")]
    en12 = [jtp(f, "request_throughput") for f in d12.glob("en_c1024_r*.json")]
    c += [
        ("içerik eşleşmeli 8/sn istek süresi oranı en az", 1.62, min(e2e), 0.005),
        ("içerik eşleşmeli 8/sn istek süresi oranı en çok", 1.68, max(e2e), 0.005),
        ("içerik eşleşmeli 12/sn TR karşılanan", 10.5, min(served), 0.05),
        ("içerik eşleşmeli 12/sn TTFT en az s", 2.5, min(ttft), 0.05),
        ("içerik eşleşmeli 12/sn TTFT en çok s", 2.7, max(ttft), 0.05),
        ("içerik eşleşmeli 12/sn EN karşılanan", 11.5, statistics.median(en12), 0.05),
    ]
    # doğal bitiş ve doğal metin
    nat = [json.loads(p.read_text(encoding="utf-8")) for p in (FU / "dogal").glob("*/ozet.json")]
    tr_diff = [x["fark_yuzde"] for n in nat for k, x in n.items() if k.startswith("tr_")]
    shift = []
    for n in nat:
        for s in ("0", "1"):
            t, e = n[f"tr_s{s}"], n[f"en_s{s}"]
            shift.append(
                abs(
                    (t["token_kelime"] / t["flores_token_kelime"])
                    / (e["token_kelime"] / e["flores_token_kelime"])
                    - 1
                )
                * 100
            )
    c += [
        ("doğal TR token/kelime farkı en az %", 1.5, min(tr_diff), 0.05),
        ("doğal TR token/kelime farkı en çok %", 8.9, max(tr_diff), 0.05),
    ]
    c.append(("doğal TR/EN kayması en fazla %", 3.8, max(shift), 0.05))
    rows = load_csv(REPO / "projects/tr-serve/results/dogal_metin_tpw.csv")
    by = {(r["model"], r["derlem"], r["dil"]): float(r["dogal_flores_orani"]) for r in rows}
    rel_n = [(by[(m, d, "tr")] / by[(m, d, "en")] - 1) * 100 for m, d, _ in by if m != "vngrs/Kumru-2B"]
    c += [
        ("doğal metin TR/EN kayması en düşük %", -8.1, min(rel_n), 0.05),
        ("doğal metin TR/EN kayması en yüksek %", 0.9, max(rel_n), 0.05),
    ]
    # ölçekli çıktı: eşli medyanlar (makaledeki sınırlama maddesi ve 1,41-1,52)
    scaled = {
        TG: (357, 1.531, 1.518),
        "gemma-4-12b-it": (349, 1.445, 1.434),
        MI: (380, 1.625, 1.517),
        QW: (365, 1.517, 1.407),
    }
    sc = []
    for m, (n, fixed, sv) in scaled.items():
        dirs = [REPO / "projects/tr-serve/results" / s / m for s in ("olcekli", "olcekli-2")]
        if m == MI:
            dirs.append(kosu / "takip-ek" / f"{MI}-olcekli")
        s_runs = paired(dirs, f"tr-o{n}", "en", 256, rho[m])
        f_runs = paired(dirs[:2], "tr", "en", 256, rho[m])
        sc.append(statistics.median(s_runs))
        c += [
            (f"ölçekli eşli {m}", sv, sc[-1], 0.0005),
            (f"aynı oturum sabit eşli {m}", fixed, statistics.median(f_runs), 0.0005),
        ]
        if m == MI:
            c.append(("Ministral ölçekli koşu sayısı", 9, len(s_runs), 0))
    c += [("ölçekli en düşük", 1.41, min(sc), 0.005), ("ölçekli en yüksek", 1.52, max(sc), 0.005)]
    return c


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--servis", type=Path, default=REPO / "projects/tr-serve/results/serve-v3")
    args = parser.parse_args()
    claims = (
        text_claims()
        + serving_claims(args.servis)
        + extra_claims(args.servis)
        + scaled_claims()
        + followup_claims(args.servis)
    )
    bad = 0
    for name, stated, actual, tol in claims:
        ok = abs(stated - actual) <= tol
        bad += not ok
        print(f"{'OK  ' if ok else 'YANLIŞ'} {name}: makale {stated}, kaynak {actual:.4f}")
    print(f"yanlış {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
