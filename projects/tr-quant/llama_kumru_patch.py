# ruff: noqa: E501  (C++ bloğundaki özgün regex satırı bölünemez)
"""llama.cpp'ye Kumru-2B ön işleyicisini ekler (b12116, 5851a8e). Tekrar çalıştırılabilir.

Kullanım: python3 llama_kumru_patch.py LLAMA_CPP_DIZINI
Düzenlenen yerler: src/llama-vocab.h (enum), src/llama-vocab.cpp (regex ve ad eşlemesi), conversion/base.py
(tokenizer hash'i). Regex'in doğruluğu kumru_regex_check.py ile (HF'e karşı) ve llama-tokenize ile denetlenir.
"""

import sys
from pathlib import Path

CHKHSH = "ddc13456fcb8e77e7d2d74c34572006552a5ec4b91bd9d2a896934e0c1a6f01c"

ENUM_ANCHOR = "    LLAMA_VOCAB_PRE_TYPE_TEKKEN_NUM3       = 66,\n"
ENUM_LINE = "    LLAMA_VOCAB_PRE_TYPE_KUMRU             = 67,\n"

REGEX_ANCHOR = "            case LLAMA_VOCAB_PRE_TYPE_GROK_2:\n"
REGEX_CASE = r"""            case LLAMA_VOCAB_PRE_TYPE_KUMRU:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "\\d|\\\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+"
                    // The letter and whitespace alternatives are double-escaped: they match literal backslash text,
                    // so letters and spaces are never split and merges span words. The last three alternatives can
                    // never win (the punctuation alternative matches first at every backslash) and are dropped.
                    // Known difference: HF \d is Unicode, here ASCII; non-ASCII decimal digits are not isolated.
                    "\\d|\\\\p\\{N\\}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|[^\\\\rnp{L}N]?\\\\p\\{L\\}+",
                };
                break;
"""

PRE_ANCHOR = '                tokenizer_pre == "grok-2") {\n'
PRE_BLOCK = """                tokenizer_pre == "kumru") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_KUMRU;
                clean_spaces = false;
            } else if (
"""

HASH_ANCHOR = '            res = "ufakzeka"\n'
HASH_BLOCK = f"""        if chkhsh == "{CHKHSH}":
            # ref: https://huggingface.co/vngrs/Kumru-2B
            res = "kumru"
"""


def insert(path: Path, anchor: str, block: str, marker: str, before: bool) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"{path.name}: zaten var")
        return
    if text.count(anchor) != 1:
        raise SystemExit(f"{path.name}: bağlantı satırı {text.count(anchor)} kez bulundu, 1 bekleniyordu")
    new = block + anchor if before else anchor + block
    path.write_text(text.replace(anchor, new), encoding="utf-8")
    print(f"{path.name}: eklendi")


def main() -> int:
    root = Path(sys.argv[1])
    insert(root / "src/llama-vocab.h", ENUM_ANCHOR, ENUM_LINE, "LLAMA_VOCAB_PRE_TYPE_KUMRU", before=False)
    insert(
        root / "src/llama-vocab.cpp",
        REGEX_ANCHOR,
        REGEX_CASE,
        "case LLAMA_VOCAB_PRE_TYPE_KUMRU:",
        before=True,
    )
    insert(root / "src/llama-vocab.cpp", PRE_ANCHOR, PRE_BLOCK, 'tokenizer_pre == "kumru"', before=True)
    insert(root / "conversion/base.py", HASH_ANCHOR, HASH_BLOCK, CHKHSH, before=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
