"""FLORES+ cümlelerini kaynak URL'ye göre belgelere toplar; TR ve EN için paralel istem dosyaları yazar.

Kullanım: python make_prompts.py --flores DIZIN --out DIZIN
Çıktı: prompts_tr.jsonl ve prompts_en.jsonl (vLLM custom dataset biçimi, alan: prompt) ve docs.json.
Her satır aynı belgenin iki dildeki halidir; talimat da o dildedir.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

INSTRUCTION = {
    "tr": "Aşağıdaki metni ayrıntılı biçimde Türkçe olarak yeniden anlat.\n\n",
    "en": "Retell the following text in detail in English.\n\n",
}
FILES = {"tr": "tur_Latn.jsonl", "en": "eng_Latn.jsonl"}


def load(flores: Path, lang: str) -> dict[int, dict]:
    rows = {}
    for split in ("dev", "devtest"):
        for line in (flores / split / FILES[lang]).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows[(split, r["id"])] = r
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tr, en = load(args.flores, "tr"), load(args.flores, "en")
    assert tr.keys() == en.keys()
    groups: dict[str, list] = defaultdict(list)
    for key in sorted(tr):
        groups[tr[key]["url"]].append(key)
    docs = []
    for url, keys in groups.items():
        docs.append(
            {
                "url": url,
                "keys": [list(k) for k in keys],
                "tr": " ".join(tr[k]["text"] for k in keys),
                "en": " ".join(en[k]["text"] for k in keys),
            }
        )
    for lang in ("tr", "en"):
        with (args.out / f"prompts_{lang}.jsonl").open("w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps({"prompt": INSTRUCTION[lang] + d[lang]}, ensure_ascii=False) + "\n")
    (args.out / "docs.json").write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = sorted(len(d["keys"]) for d in docs)
    print(
        f"belge: {len(docs)}, cümle: {len(tr)}, belge başına cümle min/medyan/max: "
        f"{sizes[0]}/{sizes[len(sizes) // 2]}/{sizes[-1]}"
    )


if __name__ == "__main__":
    main()
