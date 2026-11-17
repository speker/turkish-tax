"""Doğal bitiş kontrolü, çalışan bir OpenAI uyumlu sunucuya karşı.

562 Türkçe ve 562 İngilizce isteme düşünme kapalı, serbest uzunlukta (en fazla --max-tokens) cevap alır;
üretilen metinde token/kelime oranını, sınıra dayanan cevap oranını ve cevap dilini FLORES+ token/kelime
oranıyla karşılaştırır. Hız ölçmez.

Kullanım (h200): python3 dogal_cevap.py --ports 8301,8302 --model qwen/qwen3.8-27b --out DİZİN
Kelime tanımı token_ratio.py ile aynı (boşlukla bölme). Token sayısı sunucunun completion_tokens değeri;
EOS ile biten cevapta bitiş token'ı bu sayıya dahil olduğu için 1 çıkarılır.
"""

import argparse
import asyncio
import csv
import json
import statistics as st
from pathlib import Path

import httpx

TR_HARF = set("çğıöşüÇĞİÖŞÜ")
TR_SIK = {"ve", "bir", "bu", "için", "ile", "da", "de", "olarak", "çok", "daha"}
EN_SIK = {"the", "and", "of", "to", "in", "is", "for", "that", "with", "as"}


def dil(text: str) -> str:
    w = [x.lower().strip(".,;:!?\"'()*") for x in text.split()]
    tr, en = sum(x in TR_SIK for x in w), sum(x in EN_SIK for x in w)
    if tr > en:
        return "tr"
    if en > tr:
        return "en" if not TR_HARF & set(text) or en > 3 * tr else "karisik"
    return "belirsiz"


async def run(a, lang: str, seed: int, prompts: list[str]) -> list[dict]:
    eps = [f"http://127.0.0.1:{p}/v1/chat/completions" for p in a.ports.split(",")]
    sem = asyncio.Semaphore(a.concurrency)
    out: list[dict] = []

    async def one(client: httpx.AsyncClient, i: int, p: str) -> None:
        body = {
            "model": a.model,
            "messages": [{"role": "user", "content": p}],
            "max_tokens": a.max_tokens,
            "seed": seed * 100000 + i,
        }
        if not a.sablonsuz:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if a.temperature is not None:
            body["temperature"] = a.temperature
        async with sem:
            for _ in range(3):
                try:
                    r = await client.post(eps[i % len(eps)], json=body, timeout=900)
                    r.raise_for_status()
                    d = r.json()
                    c = d["choices"][0]
                    out.append(
                        {
                            "idx": i,
                            "text": c["message"]["content"],
                            "finish": c["finish_reason"],
                            "completion_tokens": d["usage"]["completion_tokens"],
                        }
                    )
                    return
                except (httpx.HTTPError, KeyError) as e:
                    err = str(e)[:200]
            out.append({"idx": i, "error": err})

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, i, p) for i, p in enumerate(prompts)))
    return sorted(out, key=lambda r: r["idx"])


def ozet(rows: list[dict], flores_tpw: float) -> dict:
    ok = [r for r in rows if "error" not in r]
    tam = [r for r in ok if r["finish"] == "stop"]
    toks = sum(r["completion_tokens"] - 1 for r in tam)
    words = sum(len(r["text"].split()) for r in tam)
    diller: dict[str, int] = {}
    for r in tam:
        g = dil(r["text"])
        diller[g] = diller.get(g, 0) + 1
    tpw = toks / words if words else None
    return {
        "istek": len(rows),
        "hata": len(rows) - len(ok),
        "sinira_dayanan": sum(r["finish"] == "length" for r in ok),
        "token_kelime": round(tpw, 4) if tpw else None,
        "flores_token_kelime": flores_tpw,
        "fark_yuzde": round((tpw / flores_tpw - 1) * 100, 2) if tpw else None,
        "cevap_token_medyan": st.median(r["completion_tokens"] for r in tam) if tam else None,
        "dil": diller,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", default="Qwen/Qwen3.8-27B", help="token_ratio.csv'deki model satırı")
    ap.add_argument("--prompts", type=Path, default=Path.home() / "work/tr-serve/data/prompts")
    ap.add_argument("--token-orani", type=Path, default=Path(__file__).resolve().parent / "token_ratio.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=3500)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument(
        "--sablonsuz", action="store_true", help="chat_template_kwargs gönderme (Mistral tokenizer modu)"
    )
    ap.add_argument("--temperature", type=float, help="verilmezse sunucunun varsayılan örneklemesi")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.token_orani, encoding="utf-8") as f:
        t = next(r for r in csv.DictReader(f) if r["model"] == a.repo)
    flores = {"tr": float(t["tr_token_kelime"]), "en": float(t["en_token_kelime"])}
    res = {}
    for seed in (int(s) for s in a.seeds.split(",")):
        for lang in ("tr", "en"):
            with open(a.prompts / f"prompts_{lang}.jsonl", encoding="utf-8") as f:
                prompts = [json.loads(line)["prompt"] for line in f]
            rows = asyncio.run(run(a, lang, seed, prompts))
            with open(a.out / f"{lang}_s{seed}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            res[f"{lang}_s{seed}"] = ozet(rows, flores[lang])
            print(lang, seed, res[f"{lang}_s{seed}"], flush=True)
    (a.out / "ozet.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
