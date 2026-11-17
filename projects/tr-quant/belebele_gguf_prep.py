"""belebele_gguf için soru dosyaları: belebele_fp8.py'nin düz metin şablonu ve tokenizasyonu (HF tokenizer).

python belebele_gguf_prep.py --hf MODEL/hf --data BELEBELE_DIZINI --belebele-py .../belebele_fp8.py --out DIZIN
Çıktı: sorular_<dil>.bin ve gold_<dil>.json. Metin yazılmaz, yalnız token kimlikleri.
"""

import argparse
import importlib.util
import json
import struct
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("belebele_fp8", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--belebele-py", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    bb = load_module(args.belebele_py)
    tok = AutoTokenizer.from_pretrained(args.hf)
    args.out.mkdir(parents=True, exist_ok=True)
    for lang in bb.LANGS:
        rows = bb.load(args.data, lang)
        buf = [struct.pack("<i", len(rows))]
        gold, n_single = [], 0
        for r in rows:
            base = bb.TEMPLATE[lang].format(
                p=r["flores_passage"],
                q=r["question"],
                a1=r["mc_answer1"],
                a2=r["mc_answer2"],
                a3=r["mc_answer3"],
                a4=r["mc_answer4"],
            )
            seqs = [tok.encode(base + " " + letter) for letter in bb.LETTERS]
            common = 0
            while all(len(s) > common for s in seqs) and len({s[common] for s in seqs}) == 1:
                common += 1
            common = max(common, 1)
            n_single += all(len(s) == common + 1 for s in seqs)
            buf.append(struct.pack("<ii", common, len(seqs)))
            for s in seqs:
                buf.append(struct.pack(f"<i{len(s)}i", len(s), *s))
            gold.append(int(r["correct_answer_num"]) - 1)
        (args.out / f"sorular_{lang}.bin").write_bytes(b"".join(buf))
        (args.out / f"gold_{lang}.json").write_text(json.dumps(gold), encoding="utf-8")
        print(f"{lang}: {len(rows)} soru, {n_single} tek token'da ayrışıyor, ilk dizi başı {seqs[0][:3]}")


if __name__ == "__main__":
    main()
