"""vLLM'de FP8'in Türkçe kalitesi: aynı model BF16 ve FP8, FLORES+ paralel metinde token log-olasılıkları.

vLLM imajının içinde çalışır (fp8_quality.sh başlatır).
  run:     python3 fp8_quality.py run --model /model --label bf16 [--quantization fp8] \
               --text-dir /text --out /out
  compare: python3 fp8_quality.py compare --dir /out --base bf16 --other fp8
  paired:  python3 fp8_quality.py paired --dir /out --base bf16 --other fp8   (TR ve EN farkı, eşleştirilmiş)
  chunks:  python3 fp8_quality.py chunks --dir /out --base bf16 --other fp8   (parca_<other>.csv)

run, her dil için metni 20 cümlelik parçalara böler (iki dilde aynı bölme) ve prompt_logprobs=1 ile konum
başına gerçek tokenın log-olasılığını ve en olası tokenı .npz'ye yazar. compare aynı konumlarda PPL oranını,
en olası token uyumunu ve gerçek tokenın log-olasılık farkını hesaplar; güven aralığı parçalar üzerinde
bootstrap.
Tam dağılım alınmadığı için sonuç KLD değildir.
.npz token id'lerini taşır (FLORES+ metnine geri çevrilebilir) ve yayınlanmaz. Bootstrap yalnız parça
başına toplamları kullandığı için chunks bunları id'siz yazar; compare ve paired .npz yoksa
parca_<other>.csv'den okur.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

SENTENCES_PER_CHUNK = 20
LANGS = ("tr", "en")
BOOTSTRAP = 2000


def load_chunks(path: Path) -> list[str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ["\n".join(lines[i : i + SENTENCES_PER_CHUNK]) for i in range(0, len(lines), SENTENCES_PER_CHUNK)]


def run(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        quantization=args.quantization,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        seed=0,
        enable_prefix_caching=False,
    )
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    tok = llm.get_tokenizer()
    for lang in LANGS:
        prompts = load_chunks(args.text_dir / f"flores_{lang}.txt")
        if args.add_bos:
            # Gemma 4'ün tokenizer'ı düz metne <bos> eklemiyor; <bos>'suz dağılım anlamsız (PPL ~23.000)
            from vllm.inputs import TokensPrompt

            bos = tok.bos_token_id
            encoded = [tok.encode(p) for p in prompts]
            prompts = [
                TokensPrompt(prompt_token_ids=ids if ids[:1] == [bos] else [bos, *ids]) for ids in encoded
            ]
        outputs = llm.generate(prompts, params)
        print(f"{args.label} {lang}: ilk tokenlar {outputs[0].prompt_token_ids[:3]}", flush=True)
        ids, logprob, top, doc = [], [], [], []
        for d, out in enumerate(outputs):
            for i in range(1, len(out.prompt_token_ids)):
                entry = out.prompt_logprobs[i]
                token = out.prompt_token_ids[i]
                ids.append(token)
                logprob.append(entry[token].logprob)
                top.append(next(t for t, lp in entry.items() if lp.rank == 1))
                doc.append(d)
        np.savez_compressed(
            args.out / f"{args.label}_{lang}.npz",
            ids=np.array(ids, np.int64),
            logprob=np.array(logprob, np.float64),
            top=np.array(top, np.int64),
            doc=np.array(doc, np.int32),
        )
        print(f"{args.label} {lang}: {len(outputs)} parça, {len(ids)} token", flush=True)
    # vLLM 0.33 motoru yorumlayıcı kapanırken çöküyor (Py_FatalError); sonuçlar yazıldı, süreç burada biter.
    os._exit(0)


def per_doc(base: dict, other: dict) -> dict:
    """Parça başına toplamlar; bootstrap örneği parçaları seçildiği kadar (ağırlıkla) sayar."""
    n = int(base["doc"].max()) + 1

    def total(x: np.ndarray) -> np.ndarray:
        return np.bincount(base["doc"], weights=x.astype(np.float64), minlength=n)

    return {
        "token": total(np.ones_like(base["logprob"])),
        "nll_b": total(-base["logprob"]),
        "nll_o": total(-other["logprob"]),
        "top": total((base["top"] == other["top"]).astype(np.float64)),
        "abs": total(np.abs(other["logprob"] - base["logprob"])),
    }


CHUNK_FIELDS = ("token", "nll_b", "nll_o", "top", "abs")


def load_sums(directory: Path, base: str, other: str, lang: str) -> dict:
    npz = directory / f"{base}_{lang}.npz", directory / f"{other}_{lang}.npz"
    if all(p.exists() for p in npz):
        b, o = (dict(np.load(p)) for p in npz)
        if not np.array_equal(b["ids"], o["ids"]):
            raise SystemExit(f"{lang}: iki koşunun token dizileri farklı; tokenizer aynı değil")
        return per_doc(b, o)
    with (directory / f"parca_{other}.csv").open(encoding="utf-8") as fh:
        rows = sorted((r for r in csv.DictReader(fh) if r["lang"] == lang), key=lambda r: int(r["chunk"]))
    return {k: np.array([float(r[k]) for r in rows]) for k in CHUNK_FIELDS}


def chunks(args: argparse.Namespace) -> None:
    missing = [
        p
        for lang in LANGS
        for p in (args.dir / f"{args.base}_{lang}.npz", args.dir / f"{args.other}_{lang}.npz")
    ]
    missing = [p for p in missing if not p.exists()]
    if missing:
        # parca_*.csv yalnız .npz'den üretilir; aksi hâlde load_sums yazılmakta olan dosyayı okurdu
        raise SystemExit(f"chunks .npz ister, eksik: {', '.join(p.name for p in missing)}")
    all_sums = {lang: load_sums(args.dir, args.base, args.other, lang) for lang in LANGS}
    out = args.dir / f"parca_{args.other}.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(("lang", "chunk", *CHUNK_FIELDS))
        for lang, sums in all_sums.items():
            for i in range(len(sums["token"])):
                writer.writerow((lang, i, *(repr(float(sums[k][i])) for k in CHUNK_FIELDS)))
    print(out)


def summarize(sums: dict, weights: np.ndarray) -> dict:
    tokens = (weights * sums["token"]).sum()
    nll_b = (weights * sums["nll_b"]).sum() / tokens
    nll_o = (weights * sums["nll_o"]).sum() / tokens
    return {
        "ppl_oran": math.exp(nll_o - nll_b),
        "top1_uyum": float((weights * sums["top"]).sum() / tokens),
        "ppl_base": math.exp(nll_b),
        "ppl_other": math.exp(nll_o),
        "ort_mutlak_logprob_farki": float((weights * sums["abs"]).sum() / tokens),
        "ek_nll_toplam": float((weights * (sums["nll_o"] - sums["nll_b"])).sum()),
    }


def resample(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)


def compare(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(0)
    result = {
        "base": args.base,
        "other": args.other,
        "bootstrap": BOOTSTRAP,
        "cumle_parca": SENTENCES_PER_CHUNK,
    }
    for lang in LANGS:
        sums = load_sums(args.dir, args.base, args.other, lang)
        n = len(sums["token"])
        point = summarize(sums, np.ones(n))
        boot = [summarize(sums, resample(rng, n)) for _ in range(BOOTSTRAP)]
        for key in ("ppl_oran", "top1_uyum"):
            values = np.array([b[key] for b in boot])
            point[f"{key}_ci95"] = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        point["token"] = int(sums["token"].sum())
        point["parca"] = int(n)
        result[lang] = point
    out = args.out_dir / f"ozet_{args.other}.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def paired(args: argparse.Namespace) -> None:
    """TR ve EN parçaları aynı içerik: aynı parça ağırlıklarıyla birlikte yeniden örnekleyip
    (TR PPL oranı - EN PPL oranı) farkının ve içerik başına ek kaybın TR/EN oranının güven aralığını verir."""
    rng = np.random.default_rng(0)
    sums = {lang: load_sums(args.dir, args.base, args.other, lang) for lang in LANGS}
    n = len(sums["tr"]["token"])
    if len(sums["en"]["token"]) != n:
        raise SystemExit("TR ve EN parça sayısı farklı")

    def stats(w: np.ndarray) -> tuple[float, float]:
        tr, en = summarize(sums["tr"], w), summarize(sums["en"], w)
        return tr["ppl_oran"] - en["ppl_oran"], tr["ek_nll_toplam"] / en["ek_nll_toplam"]

    point = stats(np.ones(n))
    boot = np.array([stats(resample(rng, n)) for _ in range(BOOTSTRAP)])
    result = {
        "base": args.base,
        "other": args.other,
        "fark_tr_eksi_en": point[0],
        "ci95": [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))],
        "icerik_basina_ek_kayip_tr_en": point[1],
        "icerik_ci95": [float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))],
        "bootstrap": BOOTSTRAP,
    }
    (args.out_dir / f"eslesik_{args.other}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--quantization")
    r.add_argument("--add-bos", action="store_true", help="tokenizer eklemiyorsa parça başına <bos> ekle")
    r.add_argument("--text-dir", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    for name in ("compare", "paired", "chunks"):
        c = sub.add_parser(name)
        c.add_argument("--dir", type=Path, required=True)
        c.add_argument("--base", required=True)
        c.add_argument("--other", required=True)
        c.add_argument("--out-dir", type=Path, help="JSON'ların yazılacağı dizin (varsayılan --dir)")
    args = parser.parse_args()
    if getattr(args, "out_dir", None) is None and args.cmd != "run":
        args.out_dir = args.dir
    {"run": run, "compare": compare, "paired": paired, "chunks": chunks}[args.cmd](args)


if __name__ == "__main__":
    main()
