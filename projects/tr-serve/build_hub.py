"""HF dataset paketi: servis, token, FP8 kalite ve GGUF KLD ölçümlerini İngilizce sütunlu Parquet'e yazar.

Kullanım: uv run python projects/tr-serve/build_hub.py
Çıktı: projects/tr-serve/hub/data/*.parquet (kart: projects/tr-serve/hub/README.md, elle yazılır).
Boş hücre null olur; sütun tipleri değerlerden çıkar (ör. imatrix'siz satırda calibration_seed null, tamsayı).
Metin içermez: FLORES+ ve Vikipedi cümleleri, istemler ve model çıktıları dosyalara girmez; yalnız sayılar.
"""

import csv
import json
import math
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from runs import MODELS, checkpoint_weights, run_group, served_repo

REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "projects/tr-serve/results/serve-v3"
TOKENS = REPO / "projects/tr-serve/results/token_ratio.csv"
FP8 = REPO / "projects/tr-serve/results/fp8-kalite"
QUANT = REPO / "projects/tr-quant/results"
OUT = REPO / "projects/tr-serve/hub/data"
RUN = re.compile(r"^(tr|en)_c(\d+)_r(\d+)\.json$")
# night_scaled_v3.sh: 1. oturum results/olcekli, 2. oturum (ters sıra, 64 eşzamanlılık) results/olcekli-2
SCALED = (REPO / "projects/tr-serve/results/olcekli", REPO / "projects/tr-serve/results/olcekli-2")
SCALED_RUN = re.compile(r"^(tr|en)(-o\d+)?_c(\d+)_r(\d+)\.json$")
IN_TABLE = {"turkish-gemma-9b-t1", "gemma-4-12b-it", "ministral-3-14b-bf16", "qwen3.8-27b"}
# 16-17 Kasım takip ölçümleri: koşu başına özet (istek başına listeler atılmış), dizin = oturum
TAKIP = REPO / "projects/tr-serve/results/takip-2026-11-17"
FOLLOWUP = {
    "takip": ("main", TAKIP / "kosu/takip"),
    "takip-tek": ("main", TAKIP / "kosu/takip-tek"),
    "takip-istemci": ("main", TAKIP / "kosu/takip-istemci"),
    "takip-ek": ("main", TAKIP / "kosu/takip-ek"),
    "takip-ek2": ("main", TAKIP / "kosu/takip-ek2"),
    "takip-ek3": ("main", TAKIP / "kosu/takip-ek3"),
    "tp2ara": ("main", REPO / "projects/tr-serve/results/tekrar-2026-11-17/kosu/tp2ara"),
    "yedek": ("second", REPO / "projects/tr-serve/results/tekrar-2026-11-17/kosu/yedek"),
}
FOLLOWUP_RUN = re.compile(r"^(?P<set>.+?)(?:-o(?P<out>\d+))?_c(?P<c>\d+)_r(?P<r>\d+)\.json$")
DOC_KLD = REPO / "projects/tr-quant/results/belge-kld"
BELEBELE_GGUF = REPO / "projects/tr-quant/results/belebele-gguf"
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


def model_of(name: str) -> str:
    """Takip ölçümlerinin dizin adları (ör. qwen3.8-27b-uzun-kv) için vLLM'in yüklediği repo."""
    prefix = max((p for p in MODELS if name.startswith(p)), key=len)
    return MODELS[prefix][1]


def followup_experiment(session: str, name: str) -> str:
    rules = (
        ("-acik-icerik", "open_loop_content_matched"),
        ("-acik", "open_loop"),
        ("-uzun-esit", "long_input_equal_tokens"),
        ("-uzun-kv", "long_input_kv_saturation"),
        ("-uzun", "long_input"),
        ("-olcekli", "scaled_output"),
        ("-tp2-ara", "tensor_parallel_points"),
        ("-tp", "tensor_parallel_points"),
        ("-dogal", "natural_stop"),
        ("-t0", "temperature_0"),
        ("-tek", "control_alone"),
    )
    if session == "takip-istemci":
        return "client_cpu"
    if name == "kumru-2b-uzun":  # ana kampanyadaki 4 kat istemli Kumru koşusunun tekrarı, uzun girdi değil
        return "replication"
    for suffix, exp in rules:
        if suffix in name:
            return exp
    return "replication" if session == "yedek" else "alone" if session == "takip-tek" else "other"


def followup() -> None:
    rows = []
    for session, (machine, root) in FOLLOWUP.items():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            env = (
                json.loads((d / "ortam.json").read_text(encoding="utf-8"))
                if (d / "ortam.json").exists()
                else {}
            )
            for f in sorted(d.glob("*.json")):
                m = FOLLOWUP_RUN.match(f.name)
                if not m:
                    continue
                r = json.loads(f.read_text(encoding="utf-8"))
                rate = r.get("request_rate")
                rows.append(
                    {
                        "session": session,
                        "machine": machine,
                        "config": d.name,
                        "experiment": followup_experiment(session, d.name),
                        "model": model_of(d.name),
                        "dynamic_quantization": "fp8"
                        if "--quantization fp8" in env.get("extra_args", "")
                        else "",
                        "tensor_parallel": env.get("tp"),
                        "gpus": env.get("gpus"),
                        "cpuset": env.get("cpuset"),
                        "client_cpuset": env.get("client_cpuset", ""),
                        "prompt_set": m["set"],
                        "lang": "en" if m["set"].startswith("en") else "tr",
                        "output_tokens_per_request": r["total_output_tokens"] // max(r["completed"], 1),
                        "request_rate": None
                        if rate in (None, "inf") or rate == float("inf")
                        else float(rate),
                        "concurrency": int(m["c"]),
                        "run": int(m["r"]),
                        "num_prompts": r["num_prompts"],
                        "completed": r["completed"],
                        "failed": r["failed"],
                        "total_input_tokens": r["total_input_tokens"],
                        "total_output_tokens": r["total_output_tokens"],
                        "duration_s": round(r["duration"], 3),
                        "output_tokens_per_s": round(r["output_throughput"], 2),
                        "requests_per_s": round(r["request_throughput"], 4),
                        "ttft_p50_ms": round(r["median_ttft_ms"], 2),
                        "ttft_p99_ms": round(r["p99_ttft_ms"], 2),
                        "tpot_p50_ms": round(r["median_tpot_ms"], 3),
                        "tpot_p99_ms": round(r["p99_tpot_ms"], 3),
                        "e2e_p50_ms": round(r["median_e2el_ms"], 1) if r.get("median_e2el_ms") else None,
                        "e2e_p99_ms": round(r["p99_e2el_ms"], 1) if r.get("p99_e2el_ms") else None,
                        "end_time_local": r["date"],
                    }
                )
    write("serving_followup.parquet", rows)


def fp8_rows(path: Path, unit: str, session: str) -> list[dict]:
    r = json.loads(path.read_text(encoding="utf-8"))
    pair_path = path.with_name(path.name.replace("ozet_", "eslesik_"))
    pair = json.loads(pair_path.read_text(encoding="utf-8")) if pair_path.exists() else {}
    out = []
    for lang in ("tr", "en"):
        x = r[lang]
        out.append(
            {
                "model": served_repo(path.parent.name),
                "comparison": f"{r['base']} vs {LABELS.get(r['other'], r['other'])}",
                "unit": unit,
                "session": session,
                "lang": lang,
                "tokens": x["token"],
                "units": x["parca"],
                # Gemma 4'te vLLM düz metin log-olasılıkları makul değil (BF16 PPL yüzlerce)
                "base_ppl_plausible": x["ppl_base"] < 100,
                "ppl_base": round(x["ppl_base"], 4),
                "ppl_other": round(x["ppl_other"], 4),
                "ppl_ratio": round(x["ppl_oran"], 6),
                "ppl_ratio_ci95_low": round(x["ppl_oran_ci95"][0], 6),
                "ppl_ratio_ci95_high": round(x["ppl_oran_ci95"][1], 6),
                "top1_agreement": round(x["top1_uyum"], 6),
                "paired_tr_minus_en": round(pair["fark_tr_eksi_en"], 6) if pair else "",
                "paired_ci95_low": round(pair["ci95"][0], 6) if pair else "",
                "paired_ci95_high": round(pair["ci95"][1], 6) if pair else "",
                "content_extra_loss_ratio_tr_en": round(pair["icerik_basina_ek_kayip_tr_en"], 4)
                if pair
                else "",
                "content_ratio_ci95_low": round(pair["icerik_ci95"][0], 4) if pair else "",
                "content_ratio_ci95_high": round(pair["icerik_ci95"][1], 4) if pair else "",
            }
        )
    return out


def fp8() -> None:
    """Parça (20 cümle, 5 Kasım ve 17 Kasım tekrarı) ve belge (562 belge, 17 Kasım) düzeyinde FP8 kalitesi."""
    rows = []
    sources = (
        (FP8, "chunk_20_sentences", "2026-11-05"),
        (TAKIP / "fp8-kalite-parca-tekrar", "chunk_20_sentences", "2026-11-17"),
        (TAKIP / "fp8-kalite-belge", "document", "2026-11-17"),
    )
    for root, unit, session in sources:
        for p in sorted(root.glob("*/ozet_*.json")):
            if not p.parent.name.startswith("_"):
                rows += fp8_rows(p, unit, session)
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


def scaled() -> None:
    rows = []
    for session, order in ((1, "tr, tr_scaled, en"), (2, "en, tr_scaled, tr")):
        root = SCALED[session - 1]
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(d.glob("*.json")):
                m = SCALED_RUN.match(f.name)
                if not m:
                    continue
                r = json.loads(f.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "model": served_repo(d.name),
                        "session": session,
                        "order_within_repetition": order,
                        "lang": "en" if m[1] == "en" else "tr",
                        "output_length": "scaled" if m[2] else "256",
                        "output_tokens_per_request": r["total_output_tokens"] // r["completed"],
                        "concurrency": int(m[3]),
                        "run": int(m[4]),
                        "num_prompts": r["num_prompts"],
                        "completed": r["completed"],
                        "failed": r["failed"],
                        "total_input_tokens": r["total_input_tokens"],
                        "total_output_tokens": r["total_output_tokens"],
                        "duration_s": round(r["duration"], 3),
                        "output_tokens_per_s": round(r["output_throughput"], 2),
                        "ttft_p50_ms": round(r["median_ttft_ms"], 2),
                        "ttft_p99_ms": round(r["p99_ttft_ms"], 2),
                        "tpot_p50_ms": round(r["median_tpot_ms"], 3),
                        "end_time_utc": r["date"],
                    }
                )
    write("serving_scaled.parquet", rows)


def variant_parts(variant: str) -> tuple[str, str, int | None]:
    quant, lang, seed = re.fullmatch(r"(\w+?)(?:-(tr|en|none))?(?:-s(\d+))?", variant).groups()
    imatrix = lang if lang in ("tr", "en") else "none"
    return quant, imatrix, None if imatrix == "none" else int(seed or 42)


def gguf_doc() -> None:
    """Belge düzeyinde KL (doc_kld.cpp): 562 FLORES+ belgesi, her belge kendi bağlamıyla, bütün token'lar."""
    rows = []
    for name, repo in GGUF_REPOS.items():
        for path in sorted((DOC_KLD / name).glob("*_*.csv")):
            lang, variant = path.stem.split("_", 1)
            with path.open(encoding="utf-8") as fh:
                docs = list(csv.DictReader(fh))
            tok = sum(int(d["token"]) for d in docs)
            nll_b = sum(float(d["nll_taban"]) for d in docs)
            nll_q = sum(float(d["nll_varyant"]) for d in docs)
            quant, imatrix, seed = variant_parts(variant)
            rows.append(
                {
                    "model": repo,
                    "eval_lang": lang,
                    "quant": quant,
                    "imatrix_lang": imatrix,
                    "calibration_seed": seed,
                    "documents": len(docs),
                    "tokens": tok,
                    "kld_sum": round(sum(float(d["kl_toplam"]) for d in docs), 4),
                    "kld_mean": sum(float(d["kl_toplam"]) for d in docs) / tok,
                    "same_top_token": sum(int(d["ayni_top1"]) for d in docs) / tok,
                    "ppl_bf16": round(math.exp(nll_b / tok), 4),
                    "ppl_quant": round(math.exp(nll_q / tok), 4),
                }
            )
    write("gguf_kld_doc.parquet", rows)


def belebele() -> None:
    """Belebele (facebook/belebele @7899cdf, 900 paralel soru): vLLM'de BF16 ve FP8, llama.cpp'de GGUF."""
    rows = []

    def add(engine, model, method, base, variant, lang, acc_b, acc_v, d, d_ci, t, t_ci, valid):
        quant, imatrix, _ = variant_parts(variant) if engine == "llama.cpp" else (None, None, None)
        rows.append(
            {
                "engine": engine,
                "model": model,
                "method": method,
                "base": base,
                "variant": variant,
                "quant": quant,
                "imatrix_lang": imatrix,
                "lang": lang,
                "questions": 900,
                "accuracy_base": round(acc_b, 6),
                "accuracy_variant": round(acc_v, 6),
                "delta_points": round(d, 4),
                "delta_ci95_low": round(d_ci[0], 4),
                "delta_ci95_high": round(d_ci[1], 4),
                "tr_minus_en_points": round(t, 4),
                "tr_minus_en_ci95_low": round(t_ci[0], 4),
                "tr_minus_en_ci95_high": round(t_ci[1], 4),
                # BF16 doğruluğu bir dilde 0,5'in altındaysa yöntem o modelde geçersiz (konum yanlılığı ya da
                # düşünme bloğu); satır kalır, karşılaştırmaya girmez
                "valid": valid,
            }
        )

    for path in sorted((TAKIP / "belebele-fp8").glob("*/belebele_fp8*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        method = "chat" if path.stem.endswith("-chat") else "plain"
        valid = min(r["tr"]["dogruluk_base"], r["en"]["dogruluk_base"]) >= 0.5
        for lang in ("tr", "en"):
            x = r[lang]
            add(
                "vllm",
                served_repo(path.parent.name),
                method,
                "bf16",
                "fp8_dynamic",
                lang,
                x["dogruluk_base"],
                x["dogruluk_other"],
                x["fark_puan"],
                r[f"{lang}_fark_ci95"],
                r["tr_eksi_en_puan"],
                r["tr_eksi_en_ci95"],
                valid,
            )
    for name, repo in GGUF_REPOS.items():
        path = BELEBELE_GGUF / name / "ozet.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        acc, base = r["dogruluk"], r["dogruluk"]["bf16"]
        valid = min(base.values()) >= 0.5
        for variant, d in r["bf16_farki"].items():
            for lang in ("tr", "en"):
                add(
                    "llama.cpp",
                    repo,
                    "plain",
                    "bf16_gguf",
                    variant,
                    lang,
                    base[lang],
                    acc[variant][lang],
                    d[lang]["puan"],
                    d[lang]["ci95"],
                    d["tr_eksi_en"]["puan"],
                    d["tr_eksi_en"]["ci95"],
                    valid,
                )
    write("belebele.parquet", rows)


def natural_text() -> None:
    """Vikipedi ve web metninde token/kelime (dogal_metin_tpw.py), FLORES+ değeriyle karşılaştırmalı."""
    with (REPO / "projects/tr-serve/results/dogal_metin_tpw.csv").open(encoding="utf-8") as fh:
        src = list(csv.DictReader(fh))
    rows = [
        {
            "tokenizer_repo": r["model"],
            "corpus": r["derlem"],
            "lang": r["dil"],
            "documents": int(r["belge"]),
            "words": int(r["kelime"]),
            "tokens": int(r["token"]),
            "tokens_per_word": float(r["token_kelime"]),
            "flores_tokens_per_word": float(r["flores_token_kelime"]),
            "ratio_to_flores": float(r["dogal_flores_orani"]),
        }
        for r in src
    ]
    write("natural_text_tokens.parquet", rows)


def natural_stop() -> None:
    """Doğal bitiş (dogal_cevap.py): düşünme kapalı, en fazla 3.500 token, 562 istem, iki tohum."""
    rows = []
    for path in sorted((TAKIP / "dogal").glob("*/ozet.json")):
        name = path.parent.name
        for key, x in json.loads(path.read_text(encoding="utf-8")).items():
            lang, seed = key.split("_s")
            rows.append(
                {
                    "model": model_of(name),
                    "weights": "fp8_dynamic" if "-fp8" in name else "bf16",
                    "temperature": 0.15 if name.endswith("-t015") else None,
                    "lang": lang,
                    "seed": int(seed),
                    "requests": x["istek"],
                    "errors": x["hata"],
                    "hit_length_limit": x["sinira_dayanan"],
                    "tokens_per_word": x["token_kelime"],
                    "flores_tokens_per_word": x["flores_token_kelime"],
                    "diff_vs_flores_pct": x["fark_yuzde"],
                    "median_answer_tokens": x["cevap_token_medyan"],
                }
            )
    write("natural_stop.parquet", rows)


if __name__ == "__main__":
    serving()
    tokens()
    fp8()
    gguf()
    scaled()
    followup()
    gguf_doc()
    belebele()
    natural_text()
    natural_stop()
