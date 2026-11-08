"""Model klasörlerinin ağırlık veri tipini ve niceleme ayarını yazdırır.

Kullanım: python3 model_dtype.py MODEL_DIZINI...
Repo zaten nicelenmiş geliyorsa (ör. FP8) "BF16" etiketli ölçüm aslında o biçimdedir; bunu yakalamak için.
"""

import json
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    cfg = json.loads((Path(arg) / "config.json").read_text(encoding="utf-8"))
    text = cfg.get("text_config", {})
    quant = cfg.get("quantization_config") or text.get("quantization_config")
    dtype = cfg.get("torch_dtype") or cfg.get("dtype") or text.get("torch_dtype")
    summary = (
        {k: quant.get(k) for k in ("quant_method", "fmt", "activation_scheme", "weight_block_size")}
        if quant
        else None
    )
    print(f"{Path(arg).name}: dtype={dtype} quantization={summary}")
