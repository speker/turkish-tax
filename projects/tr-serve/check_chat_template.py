"""vllm bench serve'ün (custom veri kümesi, --skip-chat-template yok) istemlere sohbet şablonu eklediğini
doğrular.

Kullanım: uv run python projects/tr-serve/check_chat_template.py
İlk 32 Türkçe istemi make_prompts.py gibi kurar, şablonsuz token sayısını v3 log'undaki total_input_tokens ile
karşılaştırır; KV cache boyutunu server.log'dan okur.
"""

import json
import pathlib
import re
from collections import defaultdict

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

V3 = pathlib.Path(__file__).parent / "results" / "serve-v3"
FL = ("openlanguagedata/flores_plus", "1147e2dc59f37fbb5c22fd8478770de02cd399f9")
INSTR = {
    "tr": "Aşağıdaki metni ayrıntılı biçimde Türkçe olarak yeniden anlat.\n\n",
    "en": "Retell the following text in detail in English.\n\n",
}
FILES = {"tr": "tur_Latn.jsonl", "en": "eng_Latn.jsonl"}


def load(lang):
    rows = {}
    for split in ("dev", "devtest"):
        p = hf_hub_download(FL[0], f"{split}/{FILES[lang]}", repo_type="dataset", revision=FL[1])
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                rows[(split, r["id"])] = r
    return rows


tr, en = load("tr"), load("en")
groups = defaultdict(list)
for k in sorted(tr):
    groups[tr[k]["url"]].append(k)
docs = [
    {"tr": " ".join(tr[k]["text"] for k in ks), "en": " ".join(en[k]["text"] for k in ks)}
    for ks in groups.values()
]
MODELS = {
    "turkish-gemma-9b-t1": ("ytu-ce-cosmos/Turkish-Gemma-9b-T1", "bc2e3064994eefab3c9043ff58c5b3c7cbfc6c40"),
    "gemma-4-12b-it": ("google/gemma-4-12B-it", "43c9d014b231474fa7212dbb48a762566c2b255d"),
    "qwen3.8-27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"),
}
for name, (repo, rev) in MODELS.items():
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=rev))
    raw = sum(len(tok.encode(INSTR["tr"] + d["tr"], add_special_tokens=False).ids) for d in docs[:32])
    logged = json.loads((V3 / name / "tr_c1_r1.json").read_text(encoding="utf-8"))["total_input_tokens"]
    print(f"{name:22s} ham {raw}  log {logged}  istem başına ek {(logged - raw) / 32:.1f}")
# Ministral: vLLM'in Mistral tokenizer'ı şablon işaretlerini düz metin olarak kodluyor
tok = Tokenizer.from_file(
    hf_hub_download(
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "tokenizer.json",
        revision="3cea74c1ebaf5ce5f5a2553de470e2ceab825142",
    )
)


def plain(text: str) -> int:
    return sum(len(tok.model.tokenize(p)) for p, _ in tok.pre_tokenizer.pre_tokenize_str(text))


for lang in ("tr", "en"):
    r = json.loads((V3 / "ministral-3-14b-bf16" / f"{lang}_c1_r1.json").read_text(encoding="utf-8"))
    logged = r["total_input_tokens"] / r["num_prompts"]
    wrapped = ["<s>[INST]" + INSTR[lang] + d[lang] + "[/INST]" for d in docs[:32]]
    control = sum(len(tok.encode(w, add_special_tokens=False).ids) for w in wrapped) / 32
    literal = sum(1 + plain(w) for w in wrapped) / 32
    print(f"ministral {lang}: log {logged:.2f}  kontrol tokenlı {control:.2f}  BOS + düz metin {literal:.2f}")
for name in ("turkish-gemma-9b-t1", "qwen3.8-27b"):
    log = (V3 / name / "server.log").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
    print(name, "KV cache tokens:", m.group(1) if m else "yok")
