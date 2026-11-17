"""Yayınlanan FP8 özetleri (ozet_*.json, eslesik_*.json) .npz olmadan, yalnız parca_*.csv'den üretilir."""

import argparse
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FP8 = ROOT / "projects/tr-serve/results/fp8-kalite"
TAKIP = ROOT / "projects/tr-serve/results/takip-2026-11-17"
spec = importlib.util.spec_from_file_location("fp8_quality", ROOT / "projects/tr-serve/fp8_quality.py")
fp8_quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp8_quality)


def close(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= 1e-9 * max(1.0, abs(a))
    return a == b


class Fp8ChunksTest(unittest.TestCase):
    def test_summaries_reproduce_from_chunk_sums(self):
        cases = sorted(FP8.glob("*/parca_*.csv"))
        self.assertGreaterEqual(len(cases), 7)
        takip = sorted(TAKIP.glob("fp8-kalite-*/*/parca_*.csv"))
        self.assertGreaterEqual(len(takip), 13)
        for csv_path in cases + takip:
            other = csv_path.stem.removeprefix("parca_")
            with self.subTest(model=csv_path.parent.name, other=other), tempfile.TemporaryDirectory() as tmp:
                want_ozet = json.loads((csv_path.parent / f"ozet_{other}.json").read_text(encoding="utf-8"))
                birim = want_ozet.get("birim", f"{fp8_quality.SENTENCES_PER_CHUNK} cümlelik parça")
                args = argparse.Namespace(
                    dir=csv_path.parent, base="bf16", other=other, out_dir=Path(tmp), birim=birim
                )
                fp8_quality.compare(args)
                outputs = [f"ozet_{other}.json"]
                if (csv_path.parent / f"eslesik_{other}.json").exists():
                    fp8_quality.paired(args)
                    outputs.append(f"eslesik_{other}.json")
                for name in outputs:
                    got = json.loads((Path(tmp) / name).read_text(encoding="utf-8"))
                    want = json.loads((csv_path.parent / name).read_text(encoding="utf-8"))
                    # birim alanından önceki özetler parça boyunu cumle_parca olarak yazar
                    if name.startswith("ozet_") and "birim" not in want:
                        got.pop("birim", None)
                        got["cumle_parca"] = fp8_quality.SENTENCES_PER_CHUNK
                    self.assertTrue(close(got, want), name)

    def test_chunks_refuses_without_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = next(FP8.glob("*/parca_fp8.csv"))
            csv_copy = Path(tmp) / "parca_fp8.csv"
            csv_copy.write_bytes(src.read_bytes())
            args = argparse.Namespace(dir=Path(tmp), base="bf16", other="fp8", out_dir=Path(tmp))
            with self.assertRaises(SystemExit):
                fp8_quality.chunks(args)
            self.assertEqual(csv_copy.read_bytes(), src.read_bytes())


if __name__ == "__main__":
    unittest.main()
