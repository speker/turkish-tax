"""Kernel against architecture for the two Gemmas (padded_backend.sh): TPOT slope per added output token in
each cell, with the backend and FlashAttention version each server logged.

Cells: Turkish-Gemma on FLASH_ATTN with FA3 (default), FA4 and TRITON_ATTN; Gemma 4 on its default
(TRITON_FLASH_ATTN, whose text queries go to FlashAttention 4) and on TRITON_ATTN. English prompts, 256 and
N = 256 x token ratio output tokens, 256 concurrent requests, five repetitions each.

    uv run python padded_backend_summary.py
"""

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).parent / "results" / "cekirdek"
CELLS = {"turkish-gemma-9b-t1": 357, "turkish-gemma-9b-t1-fa4": 357, "turkish-gemma-9b-t1-triton": 357,
         "gemma-4-12b-it": 349, "gemma-4-12b-it-triton": 349}
REPS = (1, 2, 3, 4, 5)


def backend(cell: str) -> str:
    log = (ROOT / cell / "server.log").read_text(encoding="utf-8", errors="replace")
    # automatic: "Using X attention backend out of ..."; forced: "Using AttentionBackendEnum.X backend."
    chosen = re.search(r"Using (?:AttentionBackendEnum\.)?(\w+) (?:attention )?backend", log)
    fa = re.search(r"Using FlashAttention version (\d)", log)
    return f"{chosen.group(1) if chosen else '?'}" + (f", FA{fa.group(1)}" if fa else "")


def metric(cell: str, label: str, rep: int, key: str) -> float:
    return json.loads((ROOT / cell / f"{label}_c256_r{rep}.json").read_text())[key]


def main() -> None:
    for cell, n in CELLS.items():
        if not (ROOT / cell / "server.log").exists():
            print(f"{cell}: no results")
            continue
        short = [metric(cell, "en", r, "mean_tpot_ms") for r in REPS]
        long_ = [metric(cell, f"en-o{n}", r, "mean_tpot_ms") for r in REPS]
        slopes = [1000 * (b - a) / (n - 256) for a, b in zip(short, long_, strict=True)]
        tok_s = [metric(cell, "en", r, "output_throughput") for r in REPS]
        print(f"{cell:28s} {backend(cell):22s} TPOT 256 {statistics.mean(short):6.2f} ms, {n} "
              f"{statistics.mean(long_):6.2f} ms; out tok/s at 256 {statistics.mean(tok_s):7.0f} "
              f"(sd {statistics.stdev(tok_s):.0f}); slope {statistics.mean(slopes):+6.2f} µs/token "
              f"(sd {statistics.stdev(slopes):.2f}; reps {', '.join(f'{s:+.2f}' for s in slopes)})")


if __name__ == "__main__":
    main()
