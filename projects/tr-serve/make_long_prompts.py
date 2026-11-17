"""Uzun girdi senaryosu: ardışık FLORES+ belgelerini birleştirerek paralel TR ve EN istemleri yazar.

Kullanım: python make_long_prompts.py --docs DIZIN/docs.json --out DIZIN [--target-chars 9000]
Çıktı: prompts_trlong.jsonl ve prompts_enlong.jsonl. Her satır aynı belge grubunun iki dildeki halidir; grup,
Türkçe metin target-chars karaktere ulaşana kadar ardışık belgelerden kurulur (~2.500 Qwen3.8 token'ı).
Talimatlar make_prompts.py ile aynı.
"""

import argparse
import json
from pathlib import Path

INSTRUCTION = {
    "tr": "Aşağıdaki metni ayrıntılı biçimde Türkçe olarak yeniden anlat.\n\n",
    "en": "Retell the following text in detail in English.\n\n",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-chars", type=int, default=9000)
    args = parser.parse_args()
    docs = json.loads(args.docs.read_text(encoding="utf-8"))
    groups, cur = [], []
    for d in docs:
        cur.append(d)
        if sum(len(x["tr"]) for x in cur) >= args.target_chars:
            groups.append(cur)
            cur = []
    for lang in ("tr", "en"):
        with (args.out / f"prompts_{lang}long.jsonl").open("w", encoding="utf-8") as fh:
            for g in groups:
                text = "\n\n".join(x[lang] for x in g)
                fh.write(json.dumps({"prompt": INSTRUCTION[lang] + text}, ensure_ascii=False) + "\n")
    tr_chars = sorted(sum(len(x["tr"]) for x in g) for g in groups)
    print(f"grup: {len(groups)}, grup başına belge ort.: {len(docs) / max(len(groups), 1):.1f}, "
          f"TR karakter min/medyan/max: {tr_chars[0]}/{tr_chars[len(tr_chars) // 2]}/{tr_chars[-1]}")


if __name__ == "__main__":
    main()
