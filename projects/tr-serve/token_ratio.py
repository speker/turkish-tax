"""Aynı içeriğin Türkçe ve İngilizce token sayısı: FLORES+ dev ve devtest (2.009 paralel cümle).

Kullanım: uv run python projects/tr-serve/token_ratio.py [--out projects/tr-serve/results/token_ratio.csv]
Her model, MODELS'teki sabit revision'ın tokenizer.json'ıyla ölçülür; özel tokenlar eklenmez.
tr_en_orani: aynı cümlelerin Türkçe token toplamının İngilizce token toplamına oranı.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

FLORES = ("openlanguagedata/flores_plus", "1147e2dc59f37fbb5c22fd8478770de02cd399f9")
MODELS = {
    "vngrs/Kumru-2B": "0b3cf37f657ef4013bdc99598545f5d317bd7397",
    "ytu-ce-cosmos/Turkish-Gemma-9b-T1": "bc2e3064994eefab3c9043ff58c5b3c7cbfc6c40",
    "google/gemma-4-12B-it": "43c9d014b231474fa7212dbb48a762566c2b255d",
    "mistralai/Ministral-3-14B-Instruct-2512": "29439f81c2be264d8d393273f99e7db9c0961120",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "openai/gpt-oss-120b": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    "Trendyol/Trendyol-LLM-8B-T1": "1aeda7229465d71662aacd26e6c6e8f167f683a7",
}
FIELDS = [
    "model",
    "revision",
    "cumle",
    "tr_kelime",
    "en_kelime",
    "tr_token",
    "en_token",
    "tr_token_kelime",
    "en_token_kelime",
    "tr_en_orani",
    "tr_karakter_token",
]


def flores(lang: str) -> list[str]:
    rows = []
    for split in ("dev", "devtest"):
        path = hf_hub_download(FLORES[0], f"{split}/{lang}.jsonl", repo_type="dataset", revision=FLORES[1])
        with open(path, encoding="utf-8") as fh:
            rows += sorted((json.loads(line) for line in fh), key=lambda r: r["id"])
    return [r["text"] for r in rows]


def count(tok: Tokenizer, texts: list[str]) -> int:
    return sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "token_ratio.csv")
    args = parser.parse_args()
    tr, en = flores("tur_Latn"), flores("eng_Latn")
    assert len(tr) == len(en)
    tr_words = sum(len(t.split()) for t in tr)
    en_words = sum(len(t.split()) for t in en)
    tr_chars = sum(len(t) for t in tr)
    rows = []
    for repo, rev in MODELS.items():
        tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=rev))
        t, e = count(tok, tr), count(tok, en)
        rows.append(
            {
                "model": repo,
                "revision": rev,
                "cumle": len(tr),
                "tr_kelime": tr_words,
                "en_kelime": en_words,
                "tr_token": t,
                "en_token": e,
                "tr_token_kelime": round(t / tr_words, 3),
                "en_token_kelime": round(e / en_words, 3),
                "tr_en_orani": round(t / e, 3),
                "tr_karakter_token": round(tr_chars / t, 3),
            }
        )
        print(f"{repo:42s} TR {t:7d}  EN {e:7d}  TR/EN {t / e:.3f}  TR tok/kelime {t / tr_words:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"cümle {len(tr)}, TR kelime {tr_words}, EN kelime {en_words}; yazıldı: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
