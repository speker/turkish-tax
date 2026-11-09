"""Türkçe istemle aynı token sayısına doldurulmuş İngilizce istemler (Gemma'da kalan yük çarpanının sınaması).

Kullanım: python make_padded_prompts.py PROMPTS_DIZINI AD=TOKENIZER.json [AD=TOKENIZER.json ...]
Her belge için İngilizce istemin sonuna sonraki belgelerin İngilizce metni eklenir ve token düzeyinde
Türkçe istemin token sayısına (aynı tokenizer, özel token'sız) kesilir. Sıra prompts_en.jsonl ile aynı;
vLLM istemcisinin sabit tohumlu tamamlaması bu yüzden aynı belgeleri seçer.
Çıktı: prompts_enpad-AD.jsonl. Her model için eşleşme ve en uzun istem yazdırılır.
"""

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer


def read(path: Path) -> list[str]:
    return [json.loads(line)["prompt"] for line in path.read_text(encoding="utf-8").splitlines()]


def fit(tok: Tokenizer, base: str, filler: str, target: int) -> str:
    ids = tok.encode(base + "\n\n" + filler, add_special_tokens=False).ids
    n = target
    for _ in range(20):
        text = tok.decode(ids[:n])
        got = len(tok.encode(text, add_special_tokens=False).ids)
        if got == target:
            return text
        n += target - got
    raise SystemExit(f"hedef {target} token tutturulamadı")


def main() -> int:
    root = Path(sys.argv[1])
    tr, en = read(root / "prompts_tr.jsonl"), read(root / "prompts_en.jsonl")
    assert len(tr) == len(en)
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        tok = Tokenizer.from_file(path)
        out, diffs, tr_lens = [], [], []
        for i, (t, e) in enumerate(zip(tr, en, strict=True)):
            target = len(tok.encode(t, add_special_tokens=False).ids)
            base = len(tok.encode(e, add_special_tokens=False).ids)
            if base >= target:
                text = e
            else:
                filler = " ".join(en[(i + k) % len(en)].split("\n\n", 1)[1] for k in range(1, 6))
                text = fit(tok, e, filler, target)
            diffs.append(len(tok.encode(text, add_special_tokens=False).ids) - target)
            tr_lens.append(target)
            out.append(json.dumps({"prompt": text}, ensure_ascii=False))
        (root / f"prompts_enpad-{name}.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")
        exact = sum(d == 0 for d in diffs)
        longer = sum(d > 0 for d in diffs)
        print(
            f"{name}: {exact}/{len(diffs)} belge tam eşit, {longer} belgede İngilizce zaten uzun "
            f"(en çok +{max(diffs)}); TR istem token: ort. {sum(tr_lens) / len(tr_lens):.1f}, "
            f"en uzun {max(tr_lens)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
