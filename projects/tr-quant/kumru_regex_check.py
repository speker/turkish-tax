"""Kumru-2B ön işleme regex'inin etkin eşdeğerini HF tokenizer'ına karşı doğrular.

Kumru'nun regex'inde (tokenizer.json) kelime ve boşluk alternatifleri çift kaçışlı. Oniguruma soldan ilk
eşleşmeyi aldığı için " ?[^\\s\\p{L}\\p{N}]+" ters eğik çizginin olduğu her konumda önce eşleşir; ters eğik
çizgiyle başlayan son üç alternatif hiç kazanamaz. Etkin eşdeğer ilk dört alternatiftir ve llama.cpp'nin
\\p{..} dönüşümüne takılmayacak biçimde yazılır (düz metin "\\p{N}" için kaçışlı süslü parantez).

Kullanım: uv run python projects/tr-quant/kumru_regex_check.py METİN_DOSYASI [...]
Çıkış 0: bütün satırlarda ön parçalar birebir aynı.
"""

import sys
from pathlib import Path

import regex
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO, REV = "vngrs/Kumru-2B", "0b3cf37f657ef4013bdc99598545f5d317bd7397"
EFFECTIVE = r"\d|\\p\{N\}| ?[^\s\p{L}\p{N}]+[\r\n]*|[^\\rnp{L}N]?\\p\{L\}+"
EDGE_CASES = [
    r"print('a\n')  # \t \\ \p{L} \p{N} x\p{L}}} \s\s",
    "Sayılar: ٣ ۴ 12,5 ve ¹²",
    "NBSP\u00a0burada\u2009ince\u3000geniş",
    "Satır 1\r\nSatır 2\n\n\tsekme",
    "Emoji 🙂 ve İıĞğŞş",
    "regex = r'[^\\r\\n\\p{L}\\p{N}]?\\p{L}+'",
]


def bytes_to_unicode() -> dict[int, str]:
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), 256)]
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


BYTE_MAP = bytes_to_unicode()
PATTERN = regex.compile(EFFECTIVE)


def isolated(text: str) -> list[str]:
    pieces, last = [], 0
    for m in PATTERN.finditer(text):
        if m.start() > last:
            pieces.append(text[last : m.start()])
        if m.end() > m.start():
            pieces.append(m.group(0))
        last = m.end()
    if last < len(text):
        pieces.append(text[last:])
    return ["".join(BYTE_MAP[b] for b in p.encode("utf-8")) for p in pieces]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json", revision=REV))
    lines = list(EDGE_CASES)
    for path in sys.argv[1:]:
        lines += [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    bad = 0
    for line in lines:
        hf = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str(line)]
        ours = isolated(line)
        if hf != ours:
            bad += 1
            if bad <= 5:
                print(f"FARK: {line[:80]!r}\n  hf:   {hf[:12]}\n  etkin: {ours[:12]}")
    print(f"{len(lines)} satır ({len(EDGE_CASES)} uç durum), farklı {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
