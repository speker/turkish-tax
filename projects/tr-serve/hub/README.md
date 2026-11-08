---
license: cc-by-4.0
language:
- tr
- tur
- en
pretty_name: Turkish tax - serving and quantization cost of Turkish on open LLMs
task_categories:
- text-generation
size_categories:
- n<1K
source_datasets:
- openlanguagedata/flores_plus
- wikimedia/wikipedia
annotations_creators:
- no-annotation
language_creators:
- found
tags:
- benchmark
- turkish
- vllm
- llama-cpp
- quantization
- tokenization
- fp8
- gguf
configs:
- config_name: serving_runs
  data_files: data/serving_runs.parquet
- config_name: token_ratio
  data_files: data/token_ratio.parquet
- config_name: fp8_quality
  data_files: data/fp8_quality.parquet
- config_name: gguf_kld
  data_files: data/gguf_kld.parquet
---

# Turkish tax: serving and quantization cost of Turkish on open LLMs

Measurements behind the article [The Turkish tax](https://huggingface.co/blog/serkanpeker/turkish-tax): how much more GPU time the same content costs in Turkish than in English on open models served with vLLM, how many tokens seven tokenizers spend on the same Turkish and English text, how much vLLM's FP8 changes Turkish perplexity, and how GGUF quantization and importance-matrix language affect Turkish KL divergence.

The files contain numbers only. No FLORES+ or Wikipedia sentences, prompts or model outputs are included.

## Configs

- `serving_runs`: one row per benchmark run (312 rows). vLLM 0.33.0, one H200 per model unless `tensor_parallel` says otherwise, 256 output tokens per request (`--ignore-eos`), prefix caching off, at 1, 16, 64 and 256 concurrent requests; Qwen3.8-27B also at 512 (one, two and four GPUs) and 1,024 (two and four GPUs). A first 1,024 attempt failed on the benchmark client's open-file limit and is not included. `config` is the result directory in the code repository; `run_group` says what it was for: `main` (the article's runs), `control_alone` and `control_second` (the same Turkish-Gemma configuration alone, to measure the effect of parallel runs), `warm_rerun_a` and `warm_rerun_b` (gpt-oss-120b after a full warm-up pass, two servers), `long_4x_prompts` (Kumru-2B with four times the prompts, three runs), `high_concurrency` and `high_concurrency_1024` (Qwen3.8-27B at 256 to 1,024 requests). `checkpoint_weights` is the precision of the weights in the model repository (Ministral-3-14B-Instruct-2512 ships FP8, gpt-oss-120b MXFP4, the rest BF16); `dynamic_quantization` is `fp8` when vLLM quantized BF16 weights at load time (`--quantization fp8`). `in_article_table` marks the runs of the four BF16 configurations in the article's main table; the table's FP8 speedups come from the matching `-fp8` configurations and from `ministral-3-14b` (vendor FP8).
- `token_ratio`: token counts of 2,009 parallel FLORES+ sentences (dev and devtest) for seven tokenizers at pinned revisions.
- `fp8_quality`: perplexity ratio and top-1 agreement of vLLM FP8 against BF16 on the same FLORES+ text, with 95% bootstrap intervals over 101 chunks and a paired Turkish-minus-English difference. `fp8_dynamic` is vLLM's `--quantization fp8` on BF16 weights, `fp8_vendor` the FP8 weights Mistral ships; `bf16 vs bf16_repeat` rows are a second BF16 run that reproduced every log-probability. `content_extra_loss_ratio_tr_en` is the extra log-loss from FP8 summed over the same content, Turkish divided by English (the same value on both language rows); for `fp8_vendor` the English loss is indistinguishable from zero, so its interval is meaningless. The per-chunk sums behind every interval are in the code repository (`results/fp8-kalite/*/parca_*.csv`).
- `gguf_kld`: llama.cpp (b12116) KL divergence of GGUF quants against BF16 on Wikipedia and FLORES+ text, for Turkish and English importance matrices, with four calibration samples (`calibration_seed`) for Q4_K_M; `calibration_seed` is empty for rows without an importance matrix. `ppl_bf16` comes from a separate BF16 `llama-perplexity` run on the same text. The PPL(base) that `llama-perplexity` prints in KL-divergence mode is not used: it stores base log-probabilities with a floor 16 logits below the top one, which puts it 0.85% to 1.28% below the BF16 run for Turkish-Gemma-9b-T1 (within 0.06% for the other two models). KL divergence skips vocabulary entries whose base log-probability is −16 or below, llama.cpp's own cutoff, the same for every row. A quantized model's perplexity can come out below BF16 (Turkish-Gemma-9b-T1 Q5_K_M and Q6_K on FLORES+ Turkish, −0.46% and −0.41%); KL divergence is the comparison metric.

## How it was measured

Scripts, raw per-run outputs and logs are in [speker/turkish-tax](https://github.com/speker/turkish-tax) (`projects/tr-serve/`, `projects/tr-quant/`). The machine is a 6×H200, 1 TB RAM virtual machine. Models in the serving runs ran up to five at a time on separate GPUs, each with its own 32 vCPUs; a control configuration measured alone and in parallel differed by at most 2.4%. The vLLM benchmark client applied each model's chat template (10 to 67 tokens per prompt).

## Limitations

- Serving cost per word is an estimate combining throughput under fixed 256-token outputs with FLORES+ token counts.
- gpt-oss-120b and Kumru-2B runs did not agree within 5% across repeats; they are in `serving_runs` but not in the article's table.
- FLORES+ may be in some models' training data.
- Perplexity from `prompt_logprobs` is not KL divergence.
- Gemma 4 12B perplexity from vLLM was not plausible and is left out of `fp8_quality`.

## Sources and licenses

Model revisions are in the `revision` column of `token_ratio` and pinned in the scripts. FLORES+ (CC BY-SA 4.0) and Wikipedia (CC BY-SA 3.0 and GFDL) supplied the text that was measured; their text is not redistributed here.

## Türkçe özet

Bu veri kümesi, açık dil modellerinde aynı içeriği Türkçe üretmenin İngilizceye göre ne kadar fazla GPU süresi gerektirdiğini ölçen sayıları içeriyor. Bunun yanında yedi tokenizer'ın aynı Türkçe ve İngilizce metne harcadığı token sayısı, vLLM'in FP8 kuantizasyonunun Türkçe perplexity'ye etkisi ve GGUF kuantizasyonunun ve önem matrisi (importance matrix, imatrix) dilinin Türkçe KL ıraksamasına etkisi de yer alıyor. Dosyalarda yalnızca sayılar var; FLORES+ ve Vikipedi metinleri, istemler ve model çıktıları paylaşılmadı. Servis ölçümleri H200 üzerinde, vLLM 0.33.0 ile ve önek önbelleği (prefix caching) kapalıyken yapıldı.
