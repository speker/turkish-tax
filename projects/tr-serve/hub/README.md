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
- 1K<n<10K
source_datasets:
- openlanguagedata/flores_plus
- wikimedia/wikipedia
- facebook/belebele
- HuggingFaceFW/fineweb-2
- HuggingFaceFW/fineweb
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
- config_name: serving_scaled
  data_files: data/serving_scaled.parquet
- config_name: serving_followup
  data_files: data/serving_followup.parquet
- config_name: gguf_kld_doc
  data_files: data/gguf_kld_doc.parquet
- config_name: belebele
  data_files: data/belebele.parquet
- config_name: natural_text_tokens
  data_files: data/natural_text_tokens.parquet
- config_name: natural_stop
  data_files: data/natural_stop.parquet
---

# Turkish tax: serving and quantization cost of Turkish on open LLMs

Measurements behind the article [The Turkish tax](https://huggingface.co/blog/serkanpeker/turkish-tax) ([Türkçe](https://huggingface.co/blog/serkanpeker/turkce-vergisi)): how much more GPU time the same content costs in Turkish than in English on open models served with vLLM, how many tokens seven tokenizers spend on the same Turkish and English text, how much vLLM's FP8 changes Turkish perplexity, and how GGUF quantization and importance-matrix language affect Turkish KL divergence.

A follow-up on 16 and 17 November 2026 added five-run replications on a second machine, more tensor-parallel points, long prompts, open-loop arrivals, natural-stop answers, document-level FP8 and GGUF measurements and Belebele accuracy (configs `serving_followup` to `natural_stop`).

The files contain numbers only. No FLORES+, Wikipedia, Belebele or web sentences, prompts or model outputs are included.

## Configs

- `serving_runs`: one row per benchmark run (312 rows). vLLM 0.33.0, one H200 per model unless `tensor_parallel` says otherwise, 256 output tokens per request (`--ignore-eos`), prefix caching off, at 1, 16, 64 and 256 concurrent requests; Qwen3.8-27B also at 512 (one, two and four GPUs) and 1,024 (two and four GPUs). A first 1,024 attempt failed on the benchmark client's open-file limit and is not included. `config` is the result directory in the code repository; `run_group` says what it was for: `main` (the article's runs), `control_alone` and `control_second` (the same Turkish-Gemma configuration alone, to measure the effect of parallel runs), `warm_rerun_a` and `warm_rerun_b` (gpt-oss-120b after a full warm-up pass, two servers), `long_4x_prompts` (Kumru-2B with four times the prompts, three runs), `high_concurrency` and `high_concurrency_1024` (Qwen3.8-27B at 256 to 1,024 requests). `checkpoint_weights` is the precision of the weights in the model repository (Ministral-3-14B-Instruct-2512 ships FP8, gpt-oss-120b MXFP4, the rest BF16); `dynamic_quantization` is `fp8` when vLLM quantized BF16 weights at load time (`--quantization fp8`). `in_article_table` marks the runs of the four BF16 configurations in the article's main table; the table's FP8 speedups come from the matching `-fp8` configurations and from `ministral-3-14b` (vendor FP8).
- `token_ratio`: token counts of 2,009 parallel FLORES+ sentences (dev and devtest) for seven tokenizers at pinned revisions.
- `fp8_quality`: perplexity ratio and top-1 agreement of vLLM FP8 against BF16 on the same FLORES+ text, with 95% bootstrap intervals and a paired Turkish-minus-English difference. `unit` is the bootstrap unit: `chunk_20_sentences` (101 chunks; session `2026-11-05` is the article's run, `2026-11-17` a repeat of Turkish-Gemma with the same code that reproduced it bit for bit) or `document` (the 562 FLORES+ documents of the serving runs, each scored with its own context, session `2026-11-17`). The two units can disagree: on Turkish-Gemma-9b-T1, Turkish lost more than English over chunks and less over documents. `base_ppl_plausible` is false for Gemma 4 12B, whose plain-text log-probabilities from vLLM were not plausible; those rows are not comparisons. `fp8_dynamic` is vLLM's `--quantization fp8` on BF16 weights, `fp8_vendor` the FP8 weights Mistral ships; `bf16 vs bf16_repeat` rows are a second BF16 run that reproduced every log-probability. `content_extra_loss_ratio_tr_en` is the extra log-loss from FP8 summed over the same content, Turkish divided by English (the same value on both language rows); for `fp8_vendor` the English loss is indistinguishable from zero, so its interval is meaningless. The per-chunk sums behind every interval are in the code repository (`results/fp8-kalite/*/parca_*.csv`).
- `gguf_kld`: llama.cpp (b12116) KL divergence of GGUF quants against BF16 on Wikipedia and FLORES+ text, for Turkish and English importance matrices, with four calibration samples (`calibration_seed`) for Q4_K_M; `calibration_seed` is empty for rows without an importance matrix. `ppl_bf16` comes from a separate BF16 `llama-perplexity` run on the same text. The PPL(base) that `llama-perplexity` prints in KL-divergence mode is not used: it stores base log-probabilities with a floor 16 logits below the top one, which puts it 0.85% to 1.28% below the BF16 run for Turkish-Gemma-9b-T1 (within 0.06% for the other two models). KL divergence skips vocabulary entries whose base log-probability is −16 or below, llama.cpp's own cutoff, the same for every row. A quantized model's perplexity can come out below BF16 (Turkish-Gemma-9b-T1 Q5_K_M and Q6_K on FLORES+ Turkish, −0.46% and −0.41%); KL divergence is the comparison metric.
- `serving_scaled`: a follow-up to `serving_runs` on the four table models (72 rows). English output stays at 256 tokens; Turkish runs once at 256 tokens and once at 256 × the model's TR/EN token ratio (`output_length` = `scaled`, 349 to 380 tokens in `output_tokens_per_request`), so that both languages generate the same content. All three ran on the same server in each repetition, in the order given by `order_within_repetition`: session 1 has two repetitions at 256 concurrent requests, session 2 (reversed order) two more at 256 and two at 64. With scaled Turkish output the TR/EN cost at 256 requests was 1.406 to 1.520, against 1.447 to 1.626 for the 256-token runs on the same servers. Ministral 3 14B's 1.483 here is the ratio of median throughputs over these sessions; five more scaled runs are in `serving_followup` (`experiment` = `scaled_output`), and the median over all nine runs is 1.517.
- `serving_followup`: per-run summaries of the 16 and 17 November follow-up (529 rows). `session` is the result directory, `machine` is `main` or `second`, and `experiment` says what the run tested: `replication` (the article's table models, five runs on the second machine, language order reversed in even runs), `alone` (Ministral 3 14B at 256 requests, five runs, the only model on the main machine), `tensor_parallel_points` (Qwen3.8-27B on 1, 2 and 4 GPUs at 32 to 384 requests), `long_input` and `long_input_equal_tokens` (prompts of about 2,700 Turkish tokens built from consecutive FLORES+ documents; `prompt_set` `enlongpad-*` is English padded to the Turkish token count), `long_input_kv_saturation` (the same at 256 requests, KV cache full), `open_loop` and `open_loop_content_matched` (Poisson arrivals at `request_rate`), `scaled_output`, `client_cpu` (benchmark client on the server's or on separate vCPUs), `control_alone`, `temperature_0` and `natural_stop` (Turkish-Gemma-9b-T1 controls). `output_tokens_per_request` tells fixed and content-matched outputs apart. In 6 of the 12 open-loop runs at 256 output tokens, one request out of 600 to 1,800 failed with a client-side connection error, and two in one content-matched open-loop run (`failed`); the latency columns exclude them. No closed-loop request failed.
- `gguf_kld_doc`: KL divergence of the same GGUF files against BF16, scored on the 562 FLORES+ documents of the serving runs in both languages, every token of each document with its own context (`projects/tr-quant/doc_kld.cpp`, llama.cpp b12116, CUDA backend), for 28 variants on Turkish-Gemma-9b-T1 and Trendyol-LLM-8B-T1 (four calibration samples for each language at Q4_K_M, IQ4_XS and IQ3_M) and 10 on Kumru-2B. Unlike `gguf_kld`, both languages are scored on the same content, so Turkish and English rows compare directly per unit of content (multiply by the token counts). Per-document sums are in the code repository (`projects/tr-quant/results/belge-kld/`). Absolute values depend on the backend: the CPU build of the same commit gave up to 8.5% higher KL divergence for K-quants on Kumru-2B.
- `belebele`: accuracy on Belebele (900 parallel questions, Turkish and English), scored by the log-probability of the answer letter. `engine` `vllm` compares BF16 with dynamic FP8 with a plain-text prompt and with the chat template (reasoning off); `llama.cpp` compares GGUF variants with the BF16 GGUF, plain text. Intervals are paired bootstrap over questions (2,000 for vLLM, 10,000 for llama.cpp), not adjusted for multiple comparisons. `valid` is false where BF16 accuracy was below 0.5 in either language (position bias or a reasoning block), which rules that method out for that model.
- `natural_text_tokens`: tokens per word of seven tokenizers on 5,000 Wikipedia and 5,000 web documents per language, next to the FLORES+ value.
- `natural_stop`: answers generated to completion (reasoning off, up to 3,500 tokens, 562 prompts per language, two seeds) and their tokens per word against FLORES+. Qwen3.8-27B was served with vLLM's dynamic FP8 quantization (`weights`), the other two in BF16. Ministral 3 14B ran at temperature 0.15, the value its model card recommends; at its default of 1.0 about half of its Turkish answers turned into garbled text.

## How it was measured

Scripts, raw per-run outputs and logs are in [speker/turkish-tax](https://github.com/speker/turkish-tax) (`projects/tr-serve/`, `projects/tr-quant/`). The article's runs used a 6×H200, 1 TB RAM virtual machine; the 17 November follow-up ran on the same machine after it was extended to eight H200 GPUs, and the replication on a second machine with five H200 GPUs connected through NVSwitch, with the same vLLM image and model files. Models in the serving runs ran up to five at a time on separate GPUs; each server and its benchmark client shared the same 32 vCPUs, and moving the client to 32 separate vCPUs changed throughput by 0.43% or less (`client_cpu`). A control configuration measured alone and in parallel differed by at most 2.4%. The vLLM benchmark client applied each model's chat template (10 to 67 tokens per prompt).

## Limitations

- Serving cost per word is an estimate combining throughput under fixed 256-token outputs with FLORES+ token counts. `serving_scaled` and `serving_followup` check the output-length part, long prompts and a full KV cache; tokens per word are higher in natural text than in FLORES+ (`natural_text_tokens`, `natural_stop`), which raises absolute GPU-hours per word.
- gpt-oss-120b and Kumru-2B runs did not agree within 5% across repeats; they are in `serving_runs` but not in the article's table.
- FLORES+ may be in some models' training data.
- Perplexity from `prompt_logprobs` is not KL divergence.
- Gemma 4 12B perplexity from vLLM was not plausible; its `fp8_quality` rows are flagged and not comparisons.
- Belebele accuracy is a single task scored by letter log-probability; it does not measure generation quality.

## Sources and licenses

Model revisions are in the `revision` column of `token_ratio` and pinned in the scripts. The measured text came from FLORES+ (CC BY-SA 4.0), Wikipedia (`wikimedia/wikipedia`, CC BY-SA 3.0 and GFDL), Belebele (`facebook/belebele` at `7899cdf`, CC BY-SA 4.0), FineWeb-2 (`tur_Latn`, ODC-By) and FineWeb (`sample-10BT`, ODC-By); none of their text is redistributed here, only counts and scores.

## Türkçe özet

Bu veri kümesi, açık dil modellerinde aynı içeriği Türkçe üretmenin İngilizceye göre ne kadar fazla GPU süresi gerektirdiğini ölçen sayıları içeriyor. Bunun yanında yedi tokenizer'ın aynı Türkçe ve İngilizce metne harcadığı token sayısı, vLLM'in FP8 kuantizasyonunun Türkçe perplexity'ye etkisi ve GGUF kuantizasyonunun ve önem matrisi (importance matrix, imatrix) dilinin Türkçe KL ıraksamasına etkisi de yer alıyor. Dosyalarda yalnızca sayılar var; FLORES+ ve Vikipedi metinleri, istemler ve model çıktıları paylaşılmadı. Servis ölçümleri H200 üzerinde, vLLM 0.33.0 ile ve önek önbelleği (prefix caching) kapalıyken yapıldı. `serving_scaled` tablosunda Türkçe çıktı, aynı içeriğe karşılık gelecek şekilde modelin token oranıyla uzatıldı; bu durumda 256 eşzamanlı istekte TR/EN maliyet 1,406 ile 1,520 arasında kaldı. 16 ve 17 Kasım 2026'daki ek ölçümler ayrı tablolarda: ikinci makinede beşer tekrar, daha fazla tensör paralelliği noktası, uzun istemler, açık döngü, doğal bitiş, belge düzeyinde FP8 ve GGUF ölçümleri ve Belebele doğruluğu.
