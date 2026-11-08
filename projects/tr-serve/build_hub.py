"""HF dataset paketi: servis, token, FP8 kalite ve GGUF KLD ölçümlerini İngilizce sütunlu Parquet'e yazar.

Kullanım: uv run python projects/tr-serve/build_hub.py
Çıktı: projects/tr-serve/hub/data/*.parquet (kart: projects/tr-serve/hub/README.md, elle yazılır).
Boş hücre null olur; sütun tipleri değerlerden çıkar (ör. imatrix'siz satırda calibration_seed null, tamsayı).
Metin içermez: FLORES+ ve Vikipedi cümleleri, istemler ve model çıktıları dosyalara girmez; yalnız sayılar.
"""

import csv
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from runs import checkpoint_weights, run_group, served_repo

REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "projects/tr-serve/results/serve-v3"
TOKENS = REPO / "projects/tr-serve/results/token_ratio.csv"
FP8 = REPO / "projects/tr-serve/results/fp8-kalite"
QUANT = REPO / "projects/tr-quant/results"
OUT = REPO / "projects/tr-serve/hub/data"
RUN = re.compile(r"^(tr|en)_c(\d+)_r(\d+)\.json$")
IN_TABLE = {"turkish-gemma-9b-t1", "gemma-4-12b-it", "ministral-3-14b-bf16", "qwen3.8-27b"}
# fp8_quality.py etiketleri -> veri kümesi etiketleri
LABELS = {"fp8": "fp8_dynamic", "fp8-yerel": "fp8_vendor", "bf16b": "bf16_repeat"}
GGUF_REPOS = {
    "turkish-gemma-9b-t1": "ytu-ce-cosmos/Turkish-Gemma-9b-T1",
    "trendyol-llm-8b-t1": "Trendyol/Trendyol-LLM-8B-T1",
    "kumru-2b": "vngrs/Kumru-2B",
}


def write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([{k: (None if v == "" else v) for k, v in r.items()} for r in rows])
    pq.write_table(table, OUT / name)
    print(f"{name}: {len(rows)} satır")


def serving() -> None:
    rows = []
    for d in sorted(p for p in SERVE.iterdir() if p.is_dir()):
        env = json.loads((d / "ortam.json").read_text(encoding="utf-8"))
        log = (d / "server.log").read_text(encoding="utf-8", errors="replace")
        prefix_off = "enable_prefix_caching=False" in log
        for f in sorted(d.glob("*.json")):
            m = RUN.match(f.name)
            if not m:
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            rows.append(
                {
                    "config": d.name,
                    "run_group": run_group(d.name),
                    "model": served_repo(d.name),
                    "checkpoint_weights": checkpoint_weights(d.name),
                    "dynamic_quantization": "fp8"
                    if "--quantization fp8" in env.get("extra_args", "")
                    else "",
                    "tensor_parallel": env["tp"],
                    "gpus": env["gpus"],
                    "cpuset": env["cpuset"],
                    "prefix_caching": not prefix_off,
                    "lang": m[1],
                    "concurrency": int(m[2]),
                    "run": int(m[3]),
                    "num_prompts": r["num_prompts"],
                    "completed": r["completed"],
                    "failed": r["failed"],
                    "total_input_tokens": r["total_input_tokens"],
                    "total_output_tokens": r["total_output_tokens"],
                    "duration_s": round(r["duration"], 3),
                    "output_tokens_per_s": round(r["output_throughput"], 2),
                    "ttft_p50_ms": round(r.get("p50_ttft_ms") or r["median_ttft_ms"], 2),
                    "ttft_p99_ms": round(r["p99_ttft_ms"], 2),
                    "tpot_p50_ms": round(r.get("p50_tpot_ms") or r["median_tpot_ms"], 3),
                    "end_time_utc": r["date"],
                    "in_article_table": d.name in IN_TABLE,
                }
            )
    write("serving_runs.parquet", rows)


def tokens() -> None:
    with TOKENS.open(encoding="utf-8") as fh:
        src = list(csv.DictReader(fh))
    rows = [
        {
            "tokenizer_repo": r["model"],
            "revision": r["revision"],
            "sentences": int(r["cumle"]),
            "tr_tokens": int(r["tr_token"]),
            "en_tokens": int(r["en_token"]),
            "tr_en_token_ratio": float(r["tr_en_orani"]),
            "tr_tokens_per_word": float(r["tr_token_kelime"]),
            "en_tokens_per_word": float(r["en_token_kelime"]),
        }
        for r in src
    ]
    write("token_ratio.parquet", rows)


def fp8() -> None:
    rows = []
    for p in sorted(FP8.glob("*/ozet_*.json")):
        if p.parent.name.startswith("_"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        pair_path = p.with_name(p.name.replace("ozet_", "eslesik_"))
        pair = json.loads(pair_path.read_text(encoding="utf-8")) if pair_path.exists() else {}
        for lang in ("tr", "en"):
            x = r[lang]
            rows.append(
                {
                    "model": served_repo(p.parent.name),
                    "comparison": f"{r['base']} vs {LABELS.get(r['other'], r['other'])}",
                    "lang": lang,
                    "tokens": x["token"],
                    "chunks": x["parca"],
                    "ppl_base": round(x["ppl_base"], 4),
                    "ppl_other": round(x["ppl_other"], 4),
                    "ppl_ratio": round(x["ppl_oran"], 6),
                    "ppl_ratio_ci95_low": round(x["ppl_oran_ci95"][0], 6),
                    "ppl_ratio_ci95_high": round(x["ppl_oran_ci95"][1], 6),
                    "top1_agreement": round(x["top1_uyum"], 6),
                    "paired_tr_minus_en": round(pair["fark_tr_eksi_en"], 6) if pair else "",
                    "paired_ci95_low": round(pair["ci95"][0], 6) if pair else "",
                    "paired_ci95_high": round(pair["ci95"][1], 6) if pair else "",
                    # aynı içerikte Türkçe ek log-kaybının İngilizceye oranı; iki dil için aynı değer
                    "content_extra_loss_ratio_tr_en": round(pair["icerik_basina_ek_kayip_tr_en"], 4)
                    if pair
                    else "",
                    "content_ratio_ci95_low": round(pair["icerik_ci95"][0], 4) if pair else "",
                    "content_ratio_ci95_high": round(pair["icerik_ci95"][1], 4) if pair else "",
                }
            )
    write("fp8_quality.parquet", rows)


def bf16_ppl(log: Path) -> str:
    """BF16 perplexity koşusunun sonucu.

    KLD günlüğündeki PPL(base) kullanılmaz: llama-perplexity bazı en yüksek logit'in 16 altında tabana
    sabitleyerek saklıyor (perplexity.cpp, log_softmax); Turkish-Gemma'da bu PPL(base)'i BF16 koşusundan
    %0,85–1,28 düşük veriyor (diğer iki modelde en fazla %0,06).
    """
    m = re.search(r"Final estimate: PPL = ([\d.]+)", log.read_text(encoding="utf-8", errors="replace"))
    if not m:
        raise SystemExit(f"BF16 PPL bulunamadı: {log}")
    return m[1]


def gguf() -> None:
    rows = []
    for name, repo in GGUF_REPOS.items():
        for sub, eval_set in (("", "wikipedia"), ("flores", "flores_plus"), ("flores-tohum", "flores_plus")):
            path = QUANT / name / sub / "kld.csv"
            if not path.exists():
                continue
            # tohum koşuları aynı FLORES+ metni ve aynı BF16 tabanıyla (flores/) ölçüldü
            base_dir = QUANT / name / ("flores" if sub == "flores-tohum" else sub)
            base_ppl = {lang: bf16_ppl(base_dir / f"ppl_bf16_{lang}.log") for lang in ("tr", "en")}
            with path.open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    # Q4_K_M-tr-s1: quant, imatrix dili, kalibrasyon tohumu (eki olmayan imatrix'ler tohum 42)
                    quant, lang, seed = re.fullmatch(
                        r"(\w+?)(?:-(tr|en|none))?(?:-s(\d+))?", r["varyant"]
                    ).groups()
                    imatrix = lang if lang in ("tr", "en") else "none"
                    rows.append(
                        {
                            "model": repo,
                            "eval_set": eval_set,
                            "eval_lang": r["dil"],
                            "quant": quant,
                            "imatrix_lang": imatrix,
                            "calibration_seed": None if imatrix == "none" else int(seed or 42),
                            "kld_mean": float(r["kld_mean"]),
                            "kld_median": float(r["kld_median"]),
                            "kld_p99": float(r["kld_p99"]),
                            "same_top_token": float(r["same_top_p"]),
                            "ppl_quant": float(r["ppl_q"]),
                            "ppl_bf16": float(base_ppl[r["dil"]]),
                        }
                    )
    write("gguf_kld.parquet", rows)


if __name__ == "__main__":
    serving()
    tokens()
    fp8()
    gguf()
