"""FP8'in görev doğruluğuna etkisi, Türkçe ve İngilizce aynı sorularda (Belebele, paralel 900 soru).

vLLM imajının içinde çalışır (belebele_fp8.sh başlatır).
  run:     python3 belebele_fp8.py run --model /model --label bf16 [--quantization fp8] [--add-bos] \
               --data /data --out /out
  compare: python3 belebele_fp8.py compare --dir /out --base bf16 --other fp8

Sohbet şablonu kullanılmaz (düşünen modellerde ilk token düşünce bloğu olurdu): istem düz metin, dört seçenek
harfi istemin sonuna eklenir; dört dizinin ortak önekinden sonraki token'ların log-olasılık toplamı
(prompt_logprobs) en yüksek olan cevaptır. Aynı yöntem BF16 ve FP8'de; fark yalnız ağırlıklar.
Belebele facebook/belebele @7899cdf, CC BY-SA 4.0; metin yayınlanmaz, yalnız skor.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

LANGS = {"tr": "tur_Latn", "en": "eng_Latn"}
LETTERS = ("A", "B", "C", "D")
TEMPLATE = {
    "tr": "Aşağıdaki metne göre soruyu yanıtla.\n\nMetin: {p}\n\nSoru: {q}\n"
    "A) {a1}\nB) {a2}\nC) {a3}\nD) {a4}\nCevap:",
    "en": "Answer the question based on the passage below.\n\nPassage: {p}\n\nQuestion: {q}\n"
    "A) {a1}\nB) {a2}\nC) {a3}\nD) {a4}\nAnswer:",
}
# --chat: talimatla eğitilmiş modeller düz metinde konum yanlılığı gösterebiliyor (Gemma 4 İngilizcede 900
# sorunun 841'inde "A"); sohbet şablonuyla (düşünme kapalı) harf, asistan cevabının ilk token'ları olarak
# puanlanır.
CHAT = {
    "tr": "Aşağıdaki metne göre soruyu yanıtla. Yalnızca doğru seçeneğin harfini (A, B, C veya D) yaz.\n\n"
    "Metin: {p}\n\nSoru: {q}\nA) {a1}\nB) {a2}\nC) {a3}\nD) {a4}",
    "en": "Answer the question based on the passage below. Reply with only the letter of the correct option "
    "(A, B, C or D).\n\nPassage: {p}\n\nQuestion: {q}\nA) {a1}\nB) {a2}\nC) {a3}\nD) {a4}",
}
BOOTSTRAP = 2000


def chat_prefix(tok, content: str) -> list[int]:
    msgs = [{"role": "user", "content": content}]
    for kw in ({"enable_thinking": False}, {}):
        try:
            out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw)
            break
        except TypeError:
            continue
    # Bazı tokenizer'lar liste yerine BatchEncoding (sözlük benzeri) döndürüyor
    if not isinstance(out, list):
        out = out["input_ids"]
    return list(out[0] if out and isinstance(out[0], list) else out)


def encode_plain(tok, text: str) -> list[int]:
    try:
        return tok.encode(text, add_special_tokens=False)
    except TypeError:
        return tok.encode(text)


def load(data: Path, lang: str) -> list[dict]:
    with open(data / f"{LANGS[lang]}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        quantization=args.quantization,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        seed=0,
        enable_prefix_caching=False,
    )
    tok = llm.get_tokenizer()
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    for lang in LANGS:
        rows = load(args.data, lang)
        prompts, meta = [], []
        for i, r in enumerate(rows):
            fields = {
                "p": r["flores_passage"],
                "q": r["question"],
                "a1": r["mc_answer1"],
                "a2": r["mc_answer2"],
                "a3": r["mc_answer3"],
                "a4": r["mc_answer4"],
            }
            seqs = []
            if args.chat:
                prefix = chat_prefix(tok, CHAT[lang].format(**fields))
                seqs = [prefix + encode_plain(tok, letter) for letter in LETTERS]
            else:
                base = TEMPLATE[lang].format(**fields)
                for letter in LETTERS:
                    ids = tok.encode(base + " " + letter)
                    if args.add_bos and ids[:1] != [tok.bos_token_id]:
                        ids = [tok.bos_token_id, *ids]
                    seqs.append(ids)
            common = 0
            while all(len(s) > common for s in seqs) and len({s[common] for s in seqs}) == 1:
                common += 1
            for k, ids in enumerate(seqs):
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                meta.append((i, k, max(common, 1)))
        outs = llm.generate(prompts, params)
        scores = np.full((len(rows), 4), -np.inf)
        for (i, k, start), o in zip(meta, outs, strict=True):
            ids = o.prompt_token_ids
            scores[i, k] = sum(o.prompt_logprobs[t][ids[t]].logprob for t in range(start, len(ids)))
        gold = np.array([int(r["correct_answer_num"]) - 1 for r in rows])
        np.savez_compressed(args.out / f"{args.label}_{lang}.npz", scores=scores, gold=gold)
        print(
            f"{args.label} {lang}: doğruluk {(scores.argmax(1) == gold).mean():.4f} ({len(rows)} soru)",
            flush=True,
        )
    os._exit(0)


def compare(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(0)
    res = {"base": args.base, "other": args.other, "bootstrap": BOOTSTRAP}
    correct = {}
    for lang in LANGS:
        b, o = (np.load(args.dir / f"{x}_{lang}.npz") for x in (args.base, args.other))
        cb, co = b["scores"].argmax(1) == b["gold"], o["scores"].argmax(1) == o["gold"]
        correct[lang] = (cb, co)
        res[lang] = {
            "dogruluk_base": float(cb.mean()),
            "dogruluk_other": float(co.mean()),
            "fark_puan": float((co.mean() - cb.mean()) * 100),
            "ayni_cevap": float((b["scores"].argmax(1) == o["scores"].argmax(1)).mean()),
        }
    n = len(correct["tr"][0])
    boots = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        d = {lang: (correct[lang][1][idx].mean() - correct[lang][0][idx].mean()) * 100 for lang in LANGS}
        boots.append((d["tr"], d["en"], d["tr"] - d["en"]))
    boots = np.array(boots)
    for j, key in enumerate(("tr_fark_ci95", "en_fark_ci95", "tr_eksi_en_ci95")):
        res[key] = [float(np.percentile(boots[:, j], 2.5)), float(np.percentile(boots[:, j], 97.5))]
    res["tr_eksi_en_puan"] = res["tr"]["fark_puan"] - res["en"]["fark_puan"]
    (args.dir / f"belebele_{args.other}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--quantization")
    r.add_argument("--add-bos", action="store_true")
    r.add_argument("--chat", action="store_true", help="sohbet şablonu, düşünme kapalı")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--dir", type=Path, required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--other", required=True)
    a = ap.parse_args()
    {"run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
