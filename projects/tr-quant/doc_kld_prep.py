"""doc_kld için belgeleri modelin kendi tokenizer'ıyla token'a çevirir (özel token'sız; BOS'u doc_kld ekler).

Kullanım: python doc_kld_prep.py --hf MODEL/hf --docs docs.json --out DİZİN [--limit N]
docs.json: servis ölçümünün 562 FLORES+ belgesi ({"tr": ..., "en": ...}); iki dilde aynı belgeler, aynı sıra.
"""

import argparse
import json
import struct
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    docs = json.loads(a.docs.read_text(encoding="utf-8"))
    if a.limit:
        docs = docs[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.hf)
    a.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_ilk{a.limit}" if a.limit else ""
    for lang in ("tr", "en"):
        ids = [tok.encode(d[lang], add_special_tokens=False) for d in docs]
        with open(a.out / f"tokens_{lang}{suffix}.bin", "wb") as f:
            f.write(struct.pack("<i", len(ids)))
            for x in ids:
                f.write(struct.pack(f"<i{len(x)}i", len(x), *x))
        print(lang, len(ids), "belge", sum(map(len, ids)), "token, en uzun", max(map(len, ids)))


if __name__ == "__main__":
    main()
