# Turkish tax

Code, raw results and logs behind the article "The Turkish tax: what serving Turkish costs on open LLMs"
([English](https://huggingface.co/blog/serkanpeker/turkish-tax), [Türkçe](https://huggingface.co/blog/serkanpeker/turkce-vergisi)). The measured numbers are also in the dataset
[serkanpeker/turkish-tax](https://huggingface.co/datasets/serkanpeker/turkish-tax).

## Layout

- `projects/tr-serve/`: vLLM serving throughput in Turkish and English, token ratios, FP8 quality.
  `bench.sh` runs one model; `night_quiet_v3.sh`, `gptoss_warm_v3.sh`, `tp_high_v3.sh` and `tp1024_v3.sh`
  sequenced the published runs into `~/work/tr-serve/results-v3/`, copied here with `tools/fetch_results.py`
  as `results/serve-v3/`: every run's JSON, `ortam.json` (environment) and `server.log`.
  `summarize.py` builds `ozet.csv`; `verify_serve.py` and `verify_article.py` recompute the
  article's numbers from the raw files.
- `projects/tr-quant/`: llama.cpp GGUF quantization with Turkish and English importance matrices,
  KL divergence on Wikipedia and FLORES+, and the Kumru-2B pre-tokenizer patch for llama.cpp
  (`llama_kumru_patch.py`).
- `archive/`: earlier runs (v2 had vLLM's default prefix caching on) and the scripts that sequenced them.

Comments and some file names are in Turkish.

## Checking the numbers

No GPU is needed to recompute the article's numbers from the raw files in this repository:

```
uv run python projects/tr-serve/verify_article.py     # the article's numbers (typed in the script) vs sources
uv run python projects/tr-serve/verify_serve.py --prefix-kapali   # raw bench JSONs against ozet.csv
uv run python projects/tr-quant/verify_kld.py         # GGUF KL divergence against llama-perplexity logs
uv run python projects/tr-serve/tp_latency.py         # tensor-parallel throughput, latency, GPU-hours
uv run python -m unittest discover -s tests           # includes recomputing every FP8 interval
```

The FP8 intervals are recomputed from `results/fp8-kalite/*/parca_*.csv`, per-chunk sums of the per-token
log-probabilities (`fp8_quality.py compare|paired --dir <model dir> --base bf16 --other fp8`). The per-token
arrays are not published because their token ids encode the FLORES+ text. In
`projects/tr-quant/results/*/kld.csv`, `ppl_base` is the base perplexity llama-perplexity prints in
KL-divergence mode, which is floored and biased low for Turkish-Gemma-9b-T1; use `ppl_bf16_*.log` (or the
dataset's `ppl_bf16`) for BF16 perplexity.

## Running the measurements

The scripts ran on one 6×H200 machine with vLLM 0.33.0 in Docker (`Dockerfile.bench`) and llama.cpp
b12116. Paths assume `~/work/tr-serve` and `~/work/tr-quant`. The sequencing scripts reserve GPUs through
`~/bin/lease`, a small local lock tool for sharing the machine between jobs; remove those lines or replace
them with your own scheduler.

`projects/tr-quant/setup.sh` builds llama.cpp b12116 and checks its commit;
`projects/tr-quant/results/ortak/ortam.txt` records the environment used (its output, plus the CUDA
libraries the binaries linked and the source of Turkish-Gemma's `tokenizer.model`). GGUF files are built by
`run_model.sh`; Trendyol-LLM-8B-T1 is in `queue_gguf.sh`, Kumru-2B runs through `kumru_pipeline.sh`, and
Turkish-Gemma-9b-T1, whose repository has no `tokenizer.model`, takes Gemma 2's:

```
SP_TOKENIZER_FROM=unsloth/gemma-2-9b@9145841c83add11c48155cd7bb87d8656703f997   bash run_model.sh ytu-ce-cosmos/Turkish-Gemma-9b-T1   bc2e3064994eefab3c9043ff58c5b3c7cbfc6c40 turkish-gemma-9b-t1
```

llama-quantize and llama-imatrix store the command-line paths of the importance matrix and calibration text
in GGUF metadata. Before upload, `clean_publish.sh` rewrote the published files with `clean_gguf_meta.py`,
which cuts those paths to file names, sets `general.name` and checks that tensor data stays byte-identical;
`results/*/clean_meta.jsonl` lists the changed keys, each file's size and SHA-256 and its tensor SHA-256;
`results/*/clean_load_test.txt` shows the measured and the uploaded Q4_K_M giving the same perplexity
(`clean_load_test.sh`).

## License

Code: MIT (see `LICENSE`). Results (files under `results/` except `*.patch`, and
`projects/tr-serve/hub/data/`): CC BY 4.0, see `LICENSE-results`.
`projects/tr-quant/llama_kumru_patch.py` and `results/kumru-2b/llama-cpp-kumru.patch` modify llama.cpp
(MIT, Copyright (c) 2023-2026 The ggml authors). The models and texts that were measured keep their own
licenses; no model weights, FLORES+ or Wikipedia text are included.

FP8 logs in `results/fp8-kalite/*/run_*.log` end with a crash message from vLLM 0.33's engine at interpreter
shutdown; each log prints its completion lines ("N parça, M token") before that, and `fp8_quality.py` exits
right after writing its outputs.
