"""Vikipedi'den kalibrasyon ve değerlendirme metinleri (TR ve EN), makale düzeyinde kesişmeyen.

Kullanım: python prepare_text.py --out DIZIN [--seed 42]
Her dil için ilk parquet parçasından makaleler alınır, 2.000 karakterden kısa olanlar atılır,
sabit tohumla karıştırılır; kalibrasyon ve değerlendirme makaleleri ayrık seçilir. pii_scan bulgusu olan
makale (e-posta, IP, telefon, TCKN, IBAN) seçilmez.
Çıktı: calib_{tr,en}.txt, eval_{tr,en}.txt ve seçilen makale kimlikleri (ids.json).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from pii_scan import findings  # noqa: E402

REPO = "wikimedia/wikipedia"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
CONFIGS = {"tr": "20231101.tr", "en": "20231101.en"}
TARGET_WORDS = {"calib": 150_000, "eval": 80_000}
MIN_CHARS = 2_000
MAX_CHARS = 12_000


def first_shard(config: str) -> str:
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION, token=False)
    shards = sorted(f for f in files if f.startswith(f"{config}/") and f.endswith(".parquet"))
    return shards[0]


def articles(lang: str) -> list[dict]:
    path = hf_hub_download(
        REPO, first_shard(CONFIGS[lang]), repo_type="dataset", revision=REVISION, token=False
    )
    table = pq.read_table(path, columns=["id", "title", "text"])
    rows = table.to_pylist()
    return [r for r in rows if len(r["text"]) >= MIN_CHARS]


def take(rows: list[dict], words: int) -> tuple[list[dict], list[dict]]:
    picked, total = [], 0
    for i, row in enumerate(rows):
        text = row["text"][:MAX_CHARS]
        if findings(text):
            continue
        picked.append({"id": row["id"], "title": row["title"], "text": text})
        total += len(text.split())
        if total >= words:
            return picked, rows[i + 1 :]
    raise SystemExit(f"yetersiz metin: {total} < {words}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ids = {"source": f"{REPO}@{REVISION}", "seed": args.seed}
    for lang in CONFIGS:
        rows = articles(lang)
        random.Random(args.seed).shuffle(rows)
        calib, rest = take(rows, TARGET_WORDS["calib"])
        evals, _ = take(rest, TARGET_WORDS["eval"])
        assert not {r["id"] for r in calib} & {r["id"] for r in evals}
        for split, part in (("calib", calib), ("eval", evals)):
            (args.out / f"{split}_{lang}.txt").write_text(
                "\n\n".join(r["text"] for r in part), encoding="utf-8"
            )
            ids[f"{split}_{lang}"] = [r["id"] for r in part]
            words = sum(len(r["text"].split()) for r in part)
            print(f"{split}_{lang}: {len(part)} makale, {words} kelime")
    (args.out / "ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
