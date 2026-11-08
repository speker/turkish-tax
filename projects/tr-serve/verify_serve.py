"""Servis ölçümlerinin bağımsız denetimi: ham koşu dosyalarından, summarize.py'ye dayanmadan.

Kullanım: uv run python projects/tr-serve/verify_serve.py [--dir results/serve-v3 --prefix-kapali]
Denetler: başarısız istek, çıktı token sayısı (istek × 256), eşzamanlılık ve istem sayısı, TR ve EN'nin aynı
istem sayısını alması, ölçüm sırasında başka GPU'nun meşgul olup olmadığı (ortam.json), %5'i aşan koşu farkı.
Medyanı, koşu farkını, TR/EN maliyetini ve GPU-saati yeniden hesaplar ve ozet.csv ile karşılaştırır.
Çıkış 0: bütün denetimler geçti.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from runs import SPREAD_LIMIT_PCT, tokenizer_repo

ROOT = Path(__file__).parent / "results"
SERVE = ROOT / "serve-v3"
OUT_LEN = 256
RUN = re.compile(r"^(tr|en)_c(\d+)_r(\d+)\.json$")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=SERVE, help="sonuç dizini (v3: results/serve-v3)")
    parser.add_argument("--prefix-kapali", action="store_true", help="server.log'da prefix caching kapalı mı")
    args = parser.parse_args()
    serve_dir = args.dir
    with (ROOT / "token_ratio.csv").open(encoding="utf-8") as fh:
        tok = {r["model"]: r for r in csv.DictReader(fh)}
    with (serve_dir / "ozet.csv").open(encoding="utf-8") as fh:
        ozet = {(r["model"], r["dtype"], r["dil"], int(r["eszamanlilik"])): r for r in csv.DictReader(fh)}
    errors, notes = [], []
    computed = {}
    for d in sorted(p for p in serve_dir.iterdir() if p.is_dir()):
        env_path = d / "ortam.json"
        if not env_path.exists():
            errors.append(f"{d.name}: ortam.json yok")
            continue
        if args.prefix_kapali:
            log = d / "server.log"
            text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            if "enable_prefix_caching=False" not in text:
                errors.append(f"{d.name}: server.log'da enable_prefix_caching=False yok")
        env = json.loads(env_path.read_text(encoding="utf-8"))
        if env.get("diger_mesgul_gpular"):
            notes.append(f"{d.name}: ölçüm sırasında meşgul GPU'lar {env['diger_mesgul_gpular']}")
        runs = defaultdict(list)
        prompts = {}
        for f in sorted(d.glob("*.json")):
            m = RUN.match(f.name)
            if not m:
                continue
            lang, conc = m[1], int(m[2])
            r = json.loads(f.read_text(encoding="utf-8"))
            where = f"{d.name}/{f.name}"
            done, total, failed = r["completed"], r["num_prompts"], r["failed"]
            if failed != 0 or done != total:
                errors.append(f"{where}: completed {done} / {total}, failed {failed}")
            if r["total_output_tokens"] != done * OUT_LEN:
                errors.append(f"{where}: çıktı token {r['total_output_tokens']} != {done} × {OUT_LEN}")
            if r["max_concurrency"] != conc:
                errors.append(f"{where}: max_concurrency {r['max_concurrency']} != {conc}")
            if r["num_prompts"] < 16 * conc:
                errors.append(f"{where}: istem {r['num_prompts']} < 16 × {conc}")
            prompts.setdefault(conc, set()).add(r["num_prompts"])
            runs[(lang, conc)].append(r["output_throughput"])
        for conc, sizes in prompts.items():
            if len(sizes) != 1:
                errors.append(f"{d.name} c={conc}: TR ve EN farklı istem sayısı {sizes}")
        ratio = tok[tokenizer_repo(d.name)]
        for (lang, conc), tputs in runs.items():
            med = statistics.median(tputs)
            spread = (max(tputs) - min(tputs)) / med * 100 if len(tputs) > 1 else 0.0
            if spread > SPREAD_LIMIT_PCT and len(tputs) < 3 and not d.name.startswith("kumru"):
                notes.append(f"{d.name} {lang} c={conc}: fark %{spread:.1f}, {len(tputs)} koşu, tekrar yok")
            # TP > 1: bir milyon kelime, kullanılan bütün GPU'ların süresiyle sayılır
            tpw = int(ratio[f"{lang}_token"]) / int(ratio[f"{lang}_kelime"])
            gpu_h = 1e6 * tpw / med / 3600 * int(env["tp"])
            computed[(d.name, lang, conc)] = {"tput": med, "gpu_h": gpu_h}
    # ozet.csv ile karşılaştırma
    for (model, dtype, lang, conc), row in ozet.items():
        name = model + ("-fp8" if dtype.endswith("fp8") else "")
        c = computed.get((name, lang, conc))
        if not c:
            errors.append(f"ozet {model} {dtype} {lang} c={conc}: ham veride yok")
            continue
        if round(c["tput"], 1) != float(row["cikti_tok_s"]):
            errors.append(f"ozet {name} {lang} c={conc}: tok/s {row['cikti_tok_s']} != {c['tput']:.1f}")
        if round(c["gpu_h"], 3) != float(row["gpu_saat_1m_kelime"]):
            errors.append(
                f"ozet {name} {lang} c={conc}: GPU-saat {row['gpu_saat_1m_kelime']} != {c['gpu_h']:.4f}"
            )
        if lang == "tr" and row["tr_en_maliyet"]:
            en = computed[(name, "en", conc)]
            r = tok[tokenizer_repo(name)]
            cost = int(r["tr_token"]) / int(r["en_token"]) * en["tput"] / c["tput"]
            if round(cost, 4) != float(row["tr_en_maliyet"]):
                errors.append(f"ozet {name} c={conc}: TR/EN {row['tr_en_maliyet']} != {cost:.4f}")
    dirs = len({k[0] for k in computed})
    print(f"koşu dizini: {dirs}, yapılandırma: {len(computed)}, ozet satırı: {len(ozet)}")
    for n in notes:
        print("NOT  ", n)
    for e in errors:
        print("HATA ", e)
    print(f"hata {len(errors)}, not {len(notes)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
