"""vLLM bench sonuç JSON'larını tek satırda özetler.

Kullanım: python3 runinfo.py DOSYA_YA_DA_KLASÖR...
"""

import json
import sys
from pathlib import Path


def main():
    files = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        files += sorted(p.glob("*_r*.json")) if p.is_dir() else [p]
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        print(
            f"{f.parent.name}/{f.name}: tamamlanan {d['completed']}/{d['num_prompts']}, "
            f"süre {d['duration']:.1f} sn, çıktı {d['output_throughput']:.1f} tok/s, "
            f"girdi {d['total_input_tokens']} token"
        )


if __name__ == "__main__":
    main()
