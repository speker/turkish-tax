"""Kumru GGUF'unun llama.cpp tokenizasyonunu HF tokenizer'ıyla birebir karşılaştırır.

Kullanım: python kumru_tokenize_check.py LLAMA_TOKENIZE MODEL.gguf TOKENIZER.json METİN [...]
Her dosya tek metin olarak iki tarafta da özel token eklenmeden tokenlanır.
Çıkış 0: bütün dosyalarda id'ler aynı.
"""

import ast
import subprocess
import sys

from tokenizers import Tokenizer


def llama_ids(binary: str, model: str, path: str) -> list[int]:
    out = subprocess.run(
        # --no-escape: llama-tokenize varsayılan olarak metindeki "\n" gibi kaçışları karaktere çeviriyor
        [binary, "-m", model, "-f", path, "--ids", "--no-bos", "--log-disable", "--no-escape"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    line = next(ln for ln in reversed(out.splitlines()) if ln.startswith("["))
    return ast.literal_eval(line)


def main() -> int:
    binary, model, tokenizer, *paths = sys.argv[1:]
    tok = Tokenizer.from_file(tokenizer)
    bad = 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        hf = tok.encode(text, add_special_tokens=False).ids
        ll = llama_ids(binary, model, path)
        first = next((i for i, (a, b) in enumerate(zip(hf, ll, strict=False)) if a != b), None)
        if first is None and len(hf) == len(ll):
            print(f"AYNI  {path}: {len(hf)} token")
            continue
        bad += 1
        first = min(len(hf), len(ll)) if first is None else first
        lo, hi = max(0, first - 3), first + 3
        print(f"FARK  {path}: hf {len(hf)}, llama.cpp {len(ll)}, ilk fark {first}")
        print(f"      hf    {tok.decode(hf[lo:hi])!r} {hf[lo:hi]}")
        print(f"      llama {tok.decode(ll[lo:hi])!r} {ll[lo:hi]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
