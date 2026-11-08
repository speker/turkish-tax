"""GGUF'taki token metinlerini HF tokenizer.json ile id id karşılaştırır.

Kullanım: python check_vocab.py MODEL.gguf TOKENIZER.json
Çıkış kodu 0: bütün id'ler aynı; 1: uyuşmazlık var (ilk örnekler yazdırılır).
"""

import json
import sys
from pathlib import Path

from gguf import GGUFReader


def gguf_tokens(path: str) -> list[str]:
    field = GGUFReader(path).fields["tokenizer.ggml.tokens"]
    return [bytes(field.parts[i]).decode("utf-8", errors="replace") for i in field.data]


def hf_tokens(path: str) -> dict[int, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    tokens = (
        {i: t for t, i in vocab.items()} if isinstance(vocab, dict) else dict(enumerate(v[0] for v in vocab))
    )
    tokens.update({t["id"]: t["content"] for t in data.get("added_tokens", [])})
    return tokens


def main() -> int:
    g, h = gguf_tokens(sys.argv[1]), hf_tokens(sys.argv[2])
    escaped, mismatches, padding = [], [], []
    for i in range(len(g)):
        if g[i] == h.get(i):
            continue
        # Gömme tablosu tokenizer'dan büyükse (Qwen: 151.936 ve 151.669) llama.cpp boşluğu [PADn] ile doldurur
        if i not in h and g[i] == f"[PAD{i}]":
            padding.append(i)
            continue
        # llama.cpp kullanıcı tanımlı SentencePiece tokenlarını ham metinle saklar ("▁▁" -> "  "),
        # tek baytlık tokenları da <0xNN> yazımıyla ("\t" -> "<0x09>"); ikisinde de id aynı.
        hf = h.get(i)
        if hf is not None and (
            g[i] == hf.replace("▁", " ") or (len(hf) == 1 and ord(hf) < 256 and g[i] == f"<0x{ord(hf):02X}>")
        ):
            escaped.append(i)
        else:
            mismatches.append((i, g[i], h.get(i)))
    print(
        f"gguf {len(g)} token, tokenizer.json {len(h)} token, dolgu [PADn] {len(padding)}, "
        f"yazım farkı (▁/boşluk, <0xNN>) {len(escaped)}, uyuşmazlık {len(mismatches)}"
    )
    if escaped:
        print(f"  yazım farkı id'leri: {escaped}")
    for i in (7, 8):
        if i < len(g):
            print(f"  id {i}: {g[i]!r}")
    for m in mismatches[:10]:
        print("  ", m)
    return 1 if mismatches or len(g) != len(h) + len(padding) else 0


if __name__ == "__main__":
    sys.exit(main())
