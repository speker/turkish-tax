---
license: gemma
language:
- tr
- tur
- en
library_name: gguf
base_model: ytu-ce-cosmos/Turkish-Gemma-9b-T1
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

# Turkish-Gemma-9b-T1 GGUF, Turkish importance matrix

GGUF quantizations of [ytu-ce-cosmos/Turkish-Gemma-9b-T1](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-T1) (revision `bc2e306`), made with llama.cpp b12116 (`5851a8e`). All files except Q8_0 use an importance matrix computed on Turkish Wikipedia text.

These files are modified versions of the original model weights: they were converted to GGUF and quantized. Gemma is provided under and subject to the Gemma Terms of Use found at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms). Use of these files is subject to the use restrictions in Section 3.2 of those terms, including the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). The `NOTICE` file in this repository carries the required notice, and a copy of the Gemma Terms of Use is included as `LICENSE`.

## Files

KL divergence against the BF16 GGUF on 2,009 parallel FLORES+ sentences (dev and devtest), 512-token windows:

| File | Size (GB) | KLD vs BF16, Turkish | KLD vs BF16, English | Same top token, Turkish |
|---|---|---|---|---|
| Q8_0.gguf | 9.83 | 0.0013 | 0.0007 | 98.5% |
| Q6_K-tr.gguf | 7.59 | 0.0051 | 0.0028 | 97.1% |
| Q5_K_M-tr.gguf | 6.65 | 0.0112 | 0.0068 | 95.4% |
| Q4_K_M-tr.gguf | 5.76 | 0.0358 | 0.0220 | 92.3% |
| IQ4_XS-tr.gguf | 5.18 | 0.0525 | 0.0322 | 90.8% |
| IQ3_M-tr.gguf | 4.49 | 0.1284 | 0.0781 | 85.5% |
| Q4_K_M, no importance matrix (reference, not uploaded) | 5.76 | 0.0546 | 0.0324 | 90.5% |

`imatrix_tr.gguf` is the importance matrix itself: activation statistics collected on 200 chunks of Turkish Wikipedia text; it contains no text. Example: `llama-server -m Q4_K_M-tr.gguf` (made and measured with llama.cpp b12116). Before upload, local file paths in the imatrix metadata were cut to file names and `general.name` was set; tensor data is byte-identical to the measured files (`clean_meta.jsonl` in the GitHub repository lists the SHA-256 of each file's tensors).

## Conversion notes

- The base repository has no `tokenizer.model`; the Gemma 2 SentencePiece model from `unsloth/gemma-2-9b` (revision `9145841`) was used for conversion. The GGUF vocabulary was checked against the base `tokenizer.json`: all 256,000 ids match, 31 of them differing only in notation (`▁` against a space, `<0x09>` against a tab). Ids 7 and 8 are the model's reasoning markers `<think>` and `</think>`; they are in the GGUF vocabulary.
- Calibration: 200 chunks of 512 tokens from Turkish Wikipedia (`wikimedia/wikipedia` 20231101.tr), articles disjoint from the Wikipedia evaluation text.
- Across four different calibration samples, a Turkish importance matrix gave about 3% lower mean Turkish KL divergence at Q4_K_M than an English one, in all four cases. The Turkish importance matrix gave about 34% lower Turkish KL divergence than none (0.0358 against 0.0546).

## Limitations

- KL divergence measures distance from the BF16 model on text not used for calibration, not task accuracy.
- Turkish text shows more KL divergence per token than English for every quant.
- FLORES+ may be in the base model's training data.

Measurements: [serkanpeker/turkish-tax](https://huggingface.co/datasets/serkanpeker/turkish-tax). Scripts and raw logs: [speker/turkish-tax](https://github.com/speker/turkish-tax).

## Türkçe özet

ytu-ce-cosmos/Turkish-Gemma-9b-T1 için llama.cpp GGUF dosyaları; Q8_0 dışındakiler Türkçe Vikipedi'den hesaplanan imatrix'le. Tablo, FLORES+ metninde BF16'ya göre KLD'yi gösteriyor. Dört farklı kalibrasyon örneğinde Türkçe imatrix, Q4_K_M'de Türkçe KLD'yi İngilizce imatrix'e göre ~%3, imatrix'siz kuantizasyona göre ~%34 düşürdü. Dosyalar Gemma Kullanım Şartları'na ve yasaklı kullanım politikasına tabidir; şartların bir kopyası `LICENSE` dosyasında.
