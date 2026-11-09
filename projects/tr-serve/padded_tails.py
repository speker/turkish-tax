"""Where does the output-length effect go? Mean, median and p99 TPOT/ITL/TTFT and slot occupancy per run of the
padded-prompt experiment (results/dolgu), English with 256 output tokens against English with the scaled length.

Occupancy = sum of request latencies / (duration × concurrency): the share of the run's wall time during which
all 256 slots held a request. Below 1 means ramp-up, drain or gaps.

    uv run python padded_tails.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent / "results" / "dolgu"
MODELS = {"turkish-gemma-9b-t1": 357, "gemma-4-12b-it": 349, "ministral-3-14b-bf16": 380, "qwen3.8-27b": 365}
FIELDS = ("mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
          "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms")


def run(path: Path) -> dict:
    d = json.loads(path.read_text())
    conc = d["max_concurrency"]
    row = {f: d[f] for f in FIELDS}
    row["duration_s"] = d["duration"]
    row["out_tok_s"] = d["output_throughput"]
    row["occupancy"] = sum(d["latencies"]) / (d["duration"] * conc)
    row["mean_e2el_ms"] = d["mean_e2el_ms"]
    return row


def mean2(model: str, label: str) -> dict:
    rows = [run(ROOT / model / f"{label}_c256_r{r}.json") for r in (1, 2)]
    return {k: (rows[0][k] + rows[1][k]) / 2 for k in rows[0]}


def main() -> None:
    for model, n in MODELS.items():
        short, long_ = mean2(model, "en"), mean2(model, f"en-o{n}")
        print(f"\n{model}  EN 256 → EN {n} (mean of 2 repetitions)")
        for k in ("out_tok_s", *FIELDS, "mean_e2el_ms", "duration_s", "occupancy"):
            ratio = long_[k] / short[k]
            print(f"  {k:16s} {short[k]:10.3f} → {long_[k]:10.3f}  ×{ratio:.3f}")
        # Every request has exactly N output tokens, so mean e2e = mean TTFT + (N − 1) × mean TPOT.
        for name, r, toks in (("256", short, 256), (str(n), long_, n)):
            rebuilt = r["mean_ttft_ms"] + (toks - 1) * r["mean_tpot_ms"]
            print(f"  e2e check {name:>3s}: {r['mean_e2el_ms']:.1f} vs TTFT + (N−1)·TPOT {rebuilt:.1f}; "
                  f"TTFT share {r['mean_ttft_ms'] / r['mean_e2el_ms']:.3f}; ms per output token "
                  f"{r['mean_e2el_ms'] / toks:.3f}")
        predicted = (n / 256) * (short["mean_e2el_ms"] / long_["mean_e2el_ms"]) * (long_["occupancy"] / short["occupancy"])
        print(f"  throughput ratio predicted from e2e and occupancy ×{predicted:.3f}, measured "
              f"×{long_['out_tok_s'] / short['out_tok_s']:.3f}")
        # Per output token: e2e / N = TTFT / N + TPOT × (N − 1) / N. Changes as a share of the 256-token value.
        base = short["mean_e2el_ms"] / 256
        ttft_term = (short["mean_ttft_ms"] / 256, long_["mean_ttft_ms"] / n)
        tpot_term = (short["mean_tpot_ms"] * 255 / 256, long_["mean_tpot_ms"] * (n - 1) / n)
        print(f"  per token {base:.3f} ms: TTFT term {ttft_term[0]:.3f} → {ttft_term[1]:.3f} "
              f"({(ttft_term[1] - ttft_term[0]) / base:+.2%}), TPOT term {tpot_term[0]:.3f} → {tpot_term[1]:.3f} "
              f"({(tpot_term[1] - tpot_term[0]) / base:+.2%}), occupancy ×{long_['occupancy'] / short['occupancy']:.3f}")


if __name__ == "__main__":
    main()
