"""Makalenin servis bölümündeki sayıları v3 sonuçlarından (results/serve-v3) çıkarır.

Kullanım: uv run python projects/tr-serve/article_numbers.py
Tablo ve aralıklar dört model üzerinden (koşuları her eşzamanlılıkta %5 içinde olanlar); gpt-oss-120b ve
Kumru-2B tabloda yok. Koşu farkı her yerde (max - min) / medyan.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent / "results"
V3 = ROOT / "serve-v3"
MODELS = [
    ("turkish-gemma-9b-t1", "ytu-ce-cosmos/Turkish-Gemma-9b-T1", "Turkish-Gemma-9b-T1"),
    ("gemma-4-12b-it", "google/gemma-4-12B-it", "Gemma 4 12B"),
    ("ministral-3-14b-bf16", "mistralai/Ministral-3-14B-Instruct-2512", "Ministral 3 14B"),
    ("qwen3.8-27b", "Qwen/Qwen3.8-27B", "Qwen3.8-27B"),
]
GPTOSS = ["gpt-oss-120b-sicak-a", "gpt-oss-120b-sicak-b"]
CONCS = (1, 16, 64, 256)


def load() -> tuple[dict, dict]:
    with (V3 / "ozet.csv").open(encoding="utf-8") as fh:
        o = {(r["model"], r["dtype"], r["dil"], int(r["eszamanlilik"])): r for r in csv.DictReader(fh)}
    with (ROOT / "token_ratio.csv").open(encoding="utf-8") as fh:
        tok = {r["model"]: r for r in csv.DictReader(fh)}
    return o, tok


def warm(lang: str, conc: int) -> tuple[float, list[float]]:
    vals = [
        json.loads(f.read_text(encoding="utf-8"))["output_throughput"]
        for n in GPTOSS
        for f in (V3 / n).glob(f"{lang}_c{conc}_r*.json")
    ]
    return (statistics.median(vals) if vals else float("nan")), vals


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    o, tok = load()

    def v(m, dt, lang, c, f):
        return float(o[(m, dt, lang, c)][f])

    # model -> (TR tok/s, EN tok/s, TR/EN, GPU-h, token oranı) @256; tüm eşzamanlılıklarda TR/EN
    table, allc = {}, []
    for m, repo, name in MODELS:
        ratio = int(tok[repo]["tr_token"]) / int(tok[repo]["en_token"])
        tr, en = v(m, "bf16", "tr", 256, "cikti_tok_s"), v(m, "bf16", "en", 256, "cikti_tok_s")
        gpuh = v(m, "bf16", "tr", 256, "gpu_saat_1m_kelime")
        fp8 = v(m, "fp8", "tr", 256, "cikti_tok_s") / tr
        table[name] = (tr, en, v(m, "bf16", "tr", 256, "tr_en_maliyet"), gpuh, ratio, f"{fp8:.2f}×")
        allc += [v(m, "bf16", "tr", c, "tr_en_maliyet") for c in CONCS]
    print("| Model | TR tok/s | EN tok/s | TR/EN | GPU-h | FP8 |")
    for name, (tr_, en_, cost, gpuh, _, fp8) in table.items():
        print(f"| {name} | {tr_:,.0f} | {en_:,.0f} | {cost:.2f} | {gpuh:.3f} | {fp8} |")
    costs = [x[2] for x in table.values()]
    slower = [(1 - x[0] / x[1]) * 100 for x in table.values()]
    shares = [(x[4] - 1) / (x[2] - 1) * 100 for x in table.values()]
    print(f"TR/EN 256: {min(costs):.2f}-{max(costs):.2f}")
    print(f"TR/EN tüm eşzamanlılıklar: {min(allc):.2f}-{max(allc):.2f}")
    print(f"Türkçe token hızı düşüklüğü 256 (1 - TR/EN tok/s): %{min(slower):.1f}-{max(slower):.1f}")
    print(f"token oranının payı 256, (oran - 1) / (maliyet - 1): %{min(shares):.0f}-{max(shares):.0f}")
    # istem başına ortalama giriş tokenı (sohbet şablonu dahil), çıktı 256 token sabit
    for m, _, name in MODELS:
        lens = []
        for lang in ("tr", "en"):
            j = json.loads((V3 / m / f"{lang}_c256_r1.json").read_text(encoding="utf-8"))
            lens.append(f"{lang} {j['total_input_tokens'] / j['completed']:.0f}")
        print(f"istem uzunluğu 256 ({name}): {', '.join(lens)}")
    # gpt-oss tabloda yok (koşular %5 içinde değil); sıcak turların dağılımı bilgi için
    for lang in ("tr", "en"):
        for c in CONCS:
            _, vals = warm(lang, c)
            if vals:
                spread = (max(vals) - min(vals)) / statistics.median(vals) * 100
                print(f"gpt-oss (tabloda yok) {lang} c={c}: {len(vals)} sıcak tur, fark %{spread:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
