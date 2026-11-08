import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "fp8_quality", Path(__file__).resolve().parents[1] / "projects/tr-serve/fp8_quality.py"
)
fq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fq)


def data(seed: int = 0):
    rng = np.random.default_rng(seed)
    doc = np.repeat(np.arange(5), [3, 4, 2, 5, 1])
    base = {"doc": doc, "logprob": -rng.random(len(doc)), "top": rng.integers(0, 3, len(doc))}
    other = {
        "doc": doc,
        "logprob": base["logprob"] - 0.1 * rng.random(len(doc)),
        "top": rng.integers(0, 3, len(doc)),
    }
    return base, other


class Bootstrap(unittest.TestCase):
    def test_weights_equal_explicit_duplication(self):
        base, other = data()
        sums = fq.per_doc(base, other)
        sample = np.array([0, 0, 3, 3, 3])
        got = fq.summarize(sums, np.bincount(sample, minlength=5).astype(float))
        idx = np.concatenate([np.flatnonzero(base["doc"] == d) for d in sample])
        nll_b, nll_o = -base["logprob"][idx].mean(), -other["logprob"][idx].mean()
        self.assertAlmostEqual(got["ppl_oran"], math.exp(nll_o - nll_b))
        self.assertAlmostEqual(got["top1_uyum"], float((base["top"][idx] == other["top"][idx]).mean()))

    def test_all_ones_is_point_estimate(self):
        base, other = data(1)
        got = fq.summarize(fq.per_doc(base, other), np.ones(5))
        nll_b, nll_o = -base["logprob"].mean(), -other["logprob"].mean()
        self.assertAlmostEqual(got["ppl_oran"], math.exp(nll_o - nll_b))


if __name__ == "__main__":
    unittest.main()
