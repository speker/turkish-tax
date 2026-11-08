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


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--servis", type=Path, default=REPO / "projects/tr-serve/results/serve-v3")
    args = parser.parse_args()
    claims = text_claims() + serving_claims(args.servis) + extra_claims(args.servis)
    bad = 0
    for name, stated, actual, tol in claims:
        ok = abs(stated - actual) <= tol
        bad += not ok
        print(f"{'OK  ' if ok else 'YANLIŞ'} {name}: makale {stated}, kaynak {actual:.4f}")
    print(f"yanlış {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
