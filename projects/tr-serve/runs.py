"""Servis sonuç dizini adlarının anlamı; summarize.py, verify_serve.py ve build_hub.py ortak kullanır.

Dizin adı: model öneki + koşu eki + isteğe bağlı "-fp8" (vLLM dinamik FP8), ör. qwen3.8-27b-tp2-yuksek-fp8.
"""

# Servis tablosuna giriş: her eşzamanlılıkta koşu farkı ((max - min) / medyan) bunu aşmamalı.
SPREAD_LIMIT_PCT = 5.0

# önek -> (token oranının alındığı repo, vLLM'in yüklediği repo); uzun önek önce denenir.
# Ministral BF16 reposu FP8 reposuyla aynı tokenizer'ı taşır (FLORES+ token id'leri birebir aynı).
MODELS = {
    "turkish-gemma-9b-t1": ("ytu-ce-cosmos/Turkish-Gemma-9b-T1", "ytu-ce-cosmos/Turkish-Gemma-9b-T1"),
    "gemma-4-12b-it": ("google/gemma-4-12B-it", "google/gemma-4-12B-it"),
    "ministral-3-14b-bf16": (
        "mistralai/Ministral-3-14B-Instruct-2512",
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
    ),
    "ministral-3-14b": ("mistralai/Ministral-3-14B-Instruct-2512", "mistralai/Ministral-3-14B-Instruct-2512"),
    "qwen3.8-27b": ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "openai/gpt-oss-120b"),
    "kumru-2b": ("vngrs/Kumru-2B", "vngrs/Kumru-2B"),
}
# Repoda hazır kuantize gelen ağırlıklar; ötekiler BF16 (vLLM'in dinamik FP8'i ayrıca "-fp8" ekiyle)
CHECKPOINT_WEIGHTS = {"ministral-3-14b": "fp8", "gpt-oss-120b": "mxfp4"}

# koşu eki -> veri kümesindeki run_group
GROUPS = {
    "": "main",
    "-tp2": "main",
    "-tp4": "main",
    "-tek": "control_alone",
    "-kontrol2": "control_second",
    "-sicak-a": "warm_rerun_a",
    "-sicak-b": "warm_rerun_b",
    "-uzun": "long_4x_prompts",
    "-yuksek": "high_concurrency",
    "-tp2-yuksek": "high_concurrency",
    "-tp4-yuksek": "high_concurrency",
    "-tp2-1024": "high_concurrency_1024",
    "-tp4-1024": "high_concurrency_1024",
}


def parse(name: str) -> tuple[str, str, bool]:
    """Dizin adı -> (model öneki, koşu eki, dinamik FP8 mi)."""
    fp8 = name.endswith("-fp8")
    base = name.removesuffix("-fp8")
    for prefix in sorted(MODELS, key=len, reverse=True):
        if base.startswith(prefix) and base[len(prefix) :] in GROUPS:
            return prefix, base[len(prefix) :], fp8
    raise KeyError(f"tanınmayan sonuç dizini: {name}")


def tokenizer_repo(name: str) -> str:
    return MODELS[parse(name)[0]][0]


def served_repo(name: str) -> str:
    return MODELS[parse(name)[0]][1]


def run_group(name: str) -> str:
    return GROUPS[parse(name)[1]]


def checkpoint_weights(name: str) -> str:
    return CHECKPOINT_WEIGHTS.get(parse(name)[0], "bf16")
