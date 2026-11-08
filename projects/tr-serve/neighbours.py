"""v3 paralel partisinde 256 eşzamanlı istek koşusu sırasında aynı anda çalışan öteki yapılandırma sayısı.

Kullanım: uv run python projects/tr-serve/neighbours.py
Bir yapılandırmanın zaman aralığı, bütün 256 koşularının ilk başlangıcından son bitişine kadardır.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

V3 = Path(__file__).parent / "results" / "serve-v3"
BATCH = [
    "turkish-gemma-9b-t1",
    "turkish-gemma-9b-t1-fp8",
    "ministral-3-14b",
    "gemma-4-12b-it",
    "gemma-4-12b-it-fp8",
    "ministral-3-14b-bf16",
    "ministral-3-14b-bf16-fp8",
    "qwen3.8-27b",
    "qwen3.8-27b-fp8",
]


def main() -> int:
    runs = {}
    for name in BATCH:
        for f in (V3 / name).glob("*_c256_r*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            end = datetime.strptime(r["date"], "%Y%m%d-%H%M%S")
            runs[(name, f.stem)] = (end - timedelta(seconds=r["duration"]), end)
    spans = {}
    for name in BATCH:
        own = [v for (m, _), v in runs.items() if m == name]
        spans[name] = (min(a for a, _ in own), max(b for _, b in own))
    for (name, stem), (start, end) in sorted(runs.items()):
        others = [m for m, (a, b) in spans.items() if m != name and a < end and b > start]
        print(f"{name:28s} {stem:12s} komşu {len(others)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
