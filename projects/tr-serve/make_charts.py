"""Makale grafikleri: repodaki ham sonuçlardan, İngilizce ve Türkçe etiketlerle PNG.

Kullanım: uv run python projects/tr-serve/make_charts.py
Çıktı: projects/tr-serve/hub/charts/{en,tr}_{tokens,cost,fp8,gguf}.png (veri kümesi reposuna yüklenir,
makaleler oradan bağlar). Sayılar makaledekilerle aynı kaynaklardan okunur; elle yazılmış değer yok.
"""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "projects/tr-serve/results"
QUANT = REPO / "projects/tr-quant/results"
OUT = REPO / "projects/tr-serve/hub/charts"
TR_C, EN_C, GREY = "#C8102E", "#1F3A68", "#9AA5B1"
SOURCE = {"en": "Source: serkanpeker/turkish-tax", "tr": "Kaynak: serkanpeker/turkish-tax"}

TOKENIZERS = {
    "vngrs/Kumru-2B": "Kumru-2B",
    "google/gemma-4-12B-it": "Gemma 4 12B",
    "ytu-ce-cosmos/Turkish-Gemma-9b-T1": "Turkish-Gemma-9b-T1",
    "Qwen/Qwen3.8-27B": "Qwen3.8-27B",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "mistralai/Ministral-3-14B-Instruct-2512": "Ministral 3 14B",
    "Trendyol/Trendyol-LLM-8B-T1": "Trendyol-LLM-8B-T1",
}
SERVED = {  # sonuç dizini -> (görünen ad, token_ratio.csv'deki repo, ölçekli TR çıktı uzunluğu)
    "gemma-4-12b-it": ("Gemma 4 12B", "google/gemma-4-12B-it", 349),
    "qwen3.8-27b": ("Qwen3.8-27B", "Qwen/Qwen3.8-27B", 365),
    "turkish-gemma-9b-t1": ("Turkish-Gemma-9b-T1", "ytu-ce-cosmos/Turkish-Gemma-9b-T1", 357),
    "ministral-3-14b-bf16": ("Ministral 3 14B", "mistralai/Ministral-3-14B-Instruct-2512", 380),
}
FP8_ROWS = (  # (dizin, etiket, görünen ad)
    ("turkish-gemma-9b-t1", "fp8", "Turkish-Gemma-9b-T1"),
    ("qwen3.8-27b", "fp8", "Qwen3.8-27B"),
    ("ministral-3-14b-bf16", "fp8", "Ministral 3 14B"),
    ("ministral-3-14b-bf16", "fp8-yerel", "Ministral 3 14B, vendor FP8"),
    ("kumru-2b", "fp8", "Kumru-2B"),
)
TEXT = {
    "tokens": {
        "en": (
            "Tokens for the same content, Turkish ÷ English",
            "2,009 parallel FLORES+ sentences",
            "Turkish = English",
        ),
        "tr": (
            "Aynı içerik için token sayısı, Türkçe ÷ İngilizce",
            "2.009 paralel FLORES+ cümlesi",
            "Türkçe = İngilizce",
        ),
    },
    "cost": {
        "en": (
            "GPU time for the same content, Turkish ÷ English",
            "256 concurrent requests, vLLM 0.33.0, BF16, one GPU per model",
            "Turkish output 256 tokens (main runs)",
            "Turkish output scaled to the same content (follow-up)",
        ),
        "tr": (
            "Aynı içeriğin GPU süresi, Türkçe ÷ İngilizce",
            "256 eşzamanlı istek, vLLM 0.33.0, BF16, model başına bir GPU",
            "Türkçe çıktı 256 token (ana ölçüm)",
            "Türkçe çıktı aynı içeriğe ölçekli (ek ölçüm)",
        ),
    },
    "fp8": {
        "en": (
            "Perplexity increase from FP8, FLORES+",
            "vs BF16, 95% bootstrap intervals over 101 chunks",
            "Turkish",
            "English",
            "perplexity increase (%)",
            {"Ministral 3 14B, vendor FP8": "Ministral 3 14B, vendor FP8"},
        ),
        "tr": (
            "FP8'in perplexity artışı, FLORES+",
            "BF16'ya göre, 101 parça üzerinde %95 bootstrap aralığı",
            "Türkçe",
            "İngilizce",
            "perplexity artışı (%)",
            {"Ministral 3 14B, vendor FP8": "Ministral 3 14B, Mistral FP8"},
        ),
    },
    "gguf": {
        "en": (
            "Turkish KL divergence of Q4_K_M against BF16, FLORES+",
            "llama.cpp b12116; matrix bars: mean of four calibration samples, dots: each sample",
            ("no importance matrix", "English matrix", "Turkish matrix"),
            "mean KL divergence per token (lower is better)",
        ),
        "tr": (
            "Q4_K_M'nin BF16'ya göre Türkçe KL ıraksaması, FLORES+",
            "llama.cpp b12116; matrisli çubuk: dört kalibrasyon örnekleminin ortalaması, nokta: her örneklem",
            ("önem matrisi yok", "İngilizce matris", "Türkçe matris"),
            "token başına ortalama KL ıraksaması (düşük daha iyi)",
        ),
    },
}


def fmt(x: float, lang: str, digits: int = 2) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "tr" else s


def style(ax, title: str, subtitle: str, lang: str) -> None:
    if lang == "tr":
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.set_title(f"{title}\n", loc="left", fontsize=13, fontweight="bold", color="#111")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#333", labelsize=9.5)


def save(fig, name: str, lang: str) -> None:
    fig.text(0.99, 0.01, SOURCE[lang], ha="right", va="bottom", fontsize=8, color="#777")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{lang}_{name}.png", dpi=200, facecolor="white")
    plt.close(fig)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tokens(lang: str) -> None:
    rows = {r["model"]: int(r["tr_token"]) / int(r["en_token"]) for r in read_csv(SERVE / "token_ratio.csv")}
    items = sorted(((TOKENIZERS[m], v) for m, v in rows.items()), key=lambda x: x[1])
    title, sub, ref = TEXT["tokens"][lang]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names, vals = zip(*items, strict=True)
    bars = ax.barh(names, vals, color=TR_C, height=0.6)
    ax.axvline(1, color="#333", lw=1, ls="--")
    ax.text(1.01, len(names) - 0.45, ref, fontsize=8.5, color="#333")
    for b, v in zip(bars, vals, strict=True):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, fmt(v, lang, 3) + "×", va="center", fontsize=9.5)
    ax.set_xlim(0, max(vals) * 1.15)
    style(ax, title, sub, lang)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "tokens", lang)


def cost(lang: str) -> None:
    v3 = {
        r["model"]: float(r["tr_en_maliyet"])
        for r in read_csv(SERVE / "serve-v3/ozet.csv")
        if r["dtype"] == "bf16" and r["dil"] == "tr" and r["eszamanlilik"] == "256"
    }
    ratios = {
        r["model"]: int(r["tr_token"]) / int(r["en_token"]) for r in read_csv(SERVE / "token_ratio.csv")
    }

    def tput(d: str, label: str) -> float:
        files = [f for s in ("olcekli", "olcekli-2") for f in (SERVE / s / d).glob(f"{label}_c256_r*.json")]
        return statistics.median(
            json.loads(f.read_text(encoding="utf-8"))["output_throughput"] for f in files
        )

    names, main, scaled = [], [], []
    for d, (name, repo, n) in SERVED.items():
        names.append(name)
        main.append(v3[d])
        scaled.append(ratios[repo] * tput(d, "en") / tput(d, f"tr-o{n}"))
    title, sub, l_main, l_scaled = TEXT["cost"][lang]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(names))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], main, w, color=TR_C, label=l_main)
    b2 = ax.bar([i + w / 2 for i in x], scaled, w, color=GREY, label=l_scaled)
    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.02,
                fmt(b.get_height(), lang) + "×",
                ha="center",
                fontsize=9,
            )
    ax.axhline(1, color="#333", lw=1, ls="--")
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, max(main) * 1.25)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax, title, sub, lang)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "cost", lang)


def fp8(lang: str) -> None:
    title, sub, l_tr, l_en, xlabel, rename = TEXT["fp8"][lang]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = []
    for i, (d, other, name) in enumerate(reversed(FP8_ROWS)):
        o = json.loads((SERVE / "fp8-kalite" / d / f"ozet_{other}.json").read_text(encoding="utf-8"))
        for lg, color, dy, lab in (("tr", TR_C, 0.12, l_tr), ("en", EN_C, -0.12, l_en)):
            v = (o[lg]["ppl_oran"] - 1) * 100
            lo, hi = ((c - 1) * 100 for c in o[lg]["ppl_oran_ci95"])
            ax.errorbar(
                v,
                i + dy,
                xerr=[[v - lo], [hi - v]],
                fmt="o",
                color=color,
                capsize=3,
                ms=6,
                label=lab if i == 0 else None,
            )
        labels.append(rename.get(name, name))
    ax.set_yticks(range(len(labels)), labels)
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax, title, sub, lang)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fp8", lang)


def gguf(lang: str) -> None:
    title, sub, groups, ylabel = TEXT["gguf"][lang]
    models = (("turkish-gemma-9b-t1", "Turkish-Gemma-9b-T1"), ("trendyol-llm-8b-t1", "Trendyol-LLM-8B-T1"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.26
    colors = (GREY, EN_C, TR_C)
    for mi, (d, _name) in enumerate(models):
        main = {
            (r["varyant"], r["dil"]): float(r["kld_mean"]) for r in read_csv(QUANT / d / "flores/kld.csv")
        }
        seeds = {
            (r["varyant"], r["dil"]): float(r["kld_mean"])
            for r in read_csv(QUANT / d / "flores-tohum/kld.csv")
        }
        samples = [
            [main[("Q4_K_M-none", "tr")]],
            [main[("Q4_K_M-en", "tr")]] + [seeds[(f"Q4_K_M-en-s{s}", "tr")] for s in (1, 2, 3)],
            [main[("Q4_K_M-tr", "tr")]] + [seeds[(f"Q4_K_M-tr-s{s}", "tr")] for s in (1, 2, 3)],
        ]
        for gi, vals in enumerate(samples):
            xc = mi + (gi - 1) * w
            mean = sum(vals) / len(vals)
            ax.bar(xc, mean, w * 0.9, color=colors[gi], label=groups[gi] if mi == 0 else None)
            if len(vals) > 1:
                ax.scatter([xc] * len(vals), vals, color="#111", s=10, zorder=3)
            ax.text(xc, max(vals) + 0.0012, fmt(mean, lang, 4), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(models)), [n for _, n in models])
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax, title, sub, lang)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "gguf", lang)


if __name__ == "__main__":
    for lang in ("en", "tr"):
        tokens(lang)
        cost(lang)
        fp8(lang)
        gguf(lang)
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))
