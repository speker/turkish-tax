---
license: apache-2.0
language:
- tr
- tur
- en
library_name: gguf
base_model: Trendyol/Trendyol-LLM-8B-T1
base_model_relation: quantized
datasets:
- wikimedia/wikipedia
- openlanguagedata/flores_plus
tags:
- gguf
- llama.cpp
- turkish
- imatrix
pipeline_tag: text-generation
---

# Trendyol-LLM-8B-T1 GGUF, Turkish importance matrix

GGUF quantizations of [Trendyol/Trendyol-LLM-8B-T1](https://huggingface.co/Trendyol/Trendyol-LLM-8B-T1) (revision `1aeda72`), made with llama.cpp b12116 (`5851a8e`). All files except Q8_0 use an importance matrix computed on Turkish Wikipedia text. The base model is licensed Apache-2.0; these files were converted to GGUF and quantized.

## Files

KL divergence against the BF16 GGUF on 2,009 parallel FLORES+ sentences (dev and devtest), 512-token windows:

| File | Size (GB) | KLD vs BF16, Turkish | KLD vs BF16, English | Same top token, Turkish |
|---|---|---|---|---|
| Q8_0.gguf | 8.71 | 0.0011 | 0.0007 | 98.2% |
| Q6_K-tr.gguf | 6.73 | 0.0036 | 0.0024 | 96.7% |
| Q5_K_M-tr.gguf | 5.85 | 0.0063 | 0.0043 | 95.6% |
| Q4_K_M-tr.gguf | 5.03 | 0.0185 | 0.0120 | 92.7% |
| IQ4_XS-tr.gguf | 4.56 | 0.0261 | 0.0163 | 91.8% |
| IQ3_M-tr.gguf | 3.90 | 0.0965 | 0.0864 | 84.1% |
| Q4_K_M, no importance matrix (reference, not uploaded) | 5.03 | 0.0333 | 0.0211 | 90.3% |

`imatrix_tr.gguf` is the importance matrix itself: activation statistics collected on 200 chunks of Turkish Wikipedia text; it contains no text. Example: `llama-server -m Q4_K_M-tr.gguf` (made and measured with llama.cpp b12116). Before upload, local file paths in the imatrix metadata were cut to file names and `general.name` was set; tensor data is byte-identical to the measured files (`clean_meta.jsonl` in the GitHub repository lists the SHA-256 of each file's tensors).

## Conversion notes

- The embedding table has 151,936 rows and the tokenizer 151,669 tokens (151,643 BPE entries plus 26 added special tokens); llama.cpp fills the 267 extra rows with `[PADn]` tokens. All 151,669 tokenizer ids match the GGUF vocabulary.
- Calibration: 200 chunks of 512 tokens from Turkish Wikipedia (`wikimedia/wikipedia` 20231101.tr).
- Across four calibration samples, Turkish and English importance matrices gave the same mean Turkish KL divergence at Q4_K_M (0.0217 and 0.0218), but the median was 9% to 11% lower with the Turkish one in each of the four. The calibration sample moved the mean by up to 46%. The uploaded matrix (seed 42) gave the lowest Turkish mean of the four samples (0.0185); the four-sample mean is 0.0217.
- At IQ3_M, in the one calibration sample measured (seed 42), the Turkish importance matrix gave 43% more English KL divergence on FLORES+ than the English one (0.0864 against 0.0606); Turkish-Gemma-9b-T1 showed no such difference. English-matrix files were measured (see the dataset) but are not uploaded.

## Limitations

- KL divergence measures distance from the BF16 model on text not used for calibration, not task accuracy.
- FLORES+ may be in the base model's training data.

Measurements: [serkanpeker/turkish-tax](https://huggingface.co/datasets/serkanpeker/turkish-tax). Scripts and raw logs: [speker/turkish-tax](https://github.com/speker/turkish-tax).

## Türkçe özet

Trendyol/Trendyol-LLM-8B-T1 için llama.cpp GGUF dosyaları; Q8_0 dışındakiler Türkçe Vikipedi'den hesaplanan imatrix'le. Dört farklı kalibrasyon örneğinde Q4_K_M'de Türkçe ve İngilizce imatrix'in ortalama Türkçe KLD'si aynı çıktı, medyan Türkçe imatrix'le her örnekte %9–11 düşüktü. IQ3_M'de ölçülen tek kalibrasyon örneğinde (tohum 42) Türkçe imatrix, FLORES+'ta İngilizce KLD'yi İngilizce imatrix'e göre %43 artırdı; İngilizce imatrix'li dosyalar ölçüldü ama yüklenmedi.
