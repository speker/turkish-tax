"""FLORES+ çeviri metin mi yanıltıyor? Doğal (çeviri olmayan) Türkçe ve İngilizce metinde kelime başına token.

Kullanım: uv run python projects/tr-serve/dogal_metin_tpw.py [--n 5000] [--out ÇIKTI.csv]
Her derlemden akışla ilk N belge alınır (sabit revision, belge başı en çok 6.000 karakter, çeviri hattıyla
aynı kesim); kelime ve token tanımı token_ratio.py ile aynı (boşlukla bölme, özel token'sız). Metin saklanmaz,
yalnız sayımlar yazılır. Çıktı her tokenizer ve derlem için kelime başına token ve FLORES+ değerine oranı.
"""

import argparse
import csv
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from token_ratio import MODELS  # noqa: E402

CORPORA = {
    "wikipedia": {
        "tr": ("wikimedia/wikipedia", "20231101.tr", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"),
        "en": ("wikimedia/wikipedia", "20231101.en", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"),
    },
    "web": {
        "tr": ("HuggingFaceFW/fineweb-2", "tur_Latn", "af9c13333eb981300149d5ca60a8e9d659b276b9"),
        "en": ("HuggingFaceFW/fineweb", "sample-10BT", "9bb295ddab0e05d785b879661af7260fed5140fc"),
    },
}
MAX_CHARS = 6000


def texts(repo: str, config: str, revision: str, n: int) -> list[str]:
    ds = load_dataset(repo, config, split="train", streaming=True, revision=revision)
    out = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if t:
            out.append(t[:MAX_CHARS])
        if len(out) >= n:
            break
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "dogal_metin_tpw.csv")
    a = ap.parse_args()
    with open(Path(__file__).parent / "results" / "token_ratio.csv", encoding="utf-8") as f:
        flores = {r["model"]: r for r in csv.DictReader(f)}
    data = {(c, lang): texts(*spec, a.n) for c, langs in CORPORA.items() for lang, spec in langs.items()}
    words = {k: sum(len(t.split()) for t in v) for k, v in data.items()}
    for k, v in data.items():
        print(f"{k[0]} {k[1]}: {len(v)} belge, {words[k]} kelime", flush=True)
    rows = []
    for repo, rev in MODELS.items():
        tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=rev))
        f = flores[repo]
        ftpw = {"tr": float(f["tr_token_kelime"]), "en": float(f["en_token_kelime"])}
        for (corpus, lang), v in data.items():
            n_tok = sum(len(e.ids) for e in tok.encode_batch(v, add_special_tokens=False))
            tpw = n_tok / words[(corpus, lang)]
            rows.append(
                {
                    "model": repo,
                    "derlem": corpus,
                    "dil": lang,
                    "belge": len(v),
                    "kelime": words[(corpus, lang)],
                    "token": n_tok,
                    "token_kelime": round(tpw, 4),
                    "flores_token_kelime": ftpw[lang],
                    "dogal_flores_orani": round(tpw / ftpw[lang], 4),
                }
            )
        by = {(r["derlem"], r["dil"]): r["token_kelime"] for r in rows if r["model"] == repo}
        fl_ratio = ftpw["tr"] / ftpw["en"]
        print(
            f"{repo:42s} FLORES TR/EN tpw {fl_ratio:.3f} | "
            + " | ".join(f"{c} TR/EN {by[(c, 'tr')] / by[(c, 'en')]:.3f}" for c in CORPORA),
            flush=True,
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"yazıldı: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
