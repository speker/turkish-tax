"""Kumru-2B tokenizer'ında ön işleme regex'i ve çok kelimeli tokenlar.

Kumru'nun regex'inde kelime ve boşluk alternatifleri çift kaçışlı (\\p{L}, \\s): ters eğik çizgiyle başlayan
düz metni arıyorlar, harf dizisini değil. Sonuçta ön işleyici yalnız rakamda ve noktalamada bölüyor ve BPE
boşluk aşan token öğrenebiliyor. Betik bunu ölçer: sözlükte ortasında boşluk olan token sayısı, FLORES+ TR/EN
token sayısı, sözlükte harfle yeni satırı ya da harfle tab'ı birleştiren token sayısı ve aynı BPE'nin
boşlukta bölen bir ön işleyiciyle verdiği karşı-olgusal token sayısı.
Karşı-olgusal yalnız bir örnektir, tahmin değildir: boşluk sınırıyla eğitilen bir tokenizer başka
birleştirmeler öğrenirdi.

Kullanım: uv run python projects/tr-serve/kumru_pretok.py [--out results/kumru_pretok.json]
"""

import argparse
import json
from pathlib import Path

import regex
from huggingface_hub import hf_hub_download
from tokenizers import Regex, Tokenizer, pre_tokenizers

REPO, REV = "vngrs/Kumru-2B", "0b3cf37f657ef4013bdc99598545f5d317bd7397"
FLORES = ("openlanguagedata/flores_plus", "1147e2dc59f37fbb5c22fd8478770de02cd399f9")
# Kelime, sayı, noktalama, boşluk ayrı; Kumru'nun tek rakam bölmesi korunur. Boşluğun kelimenin önüne mi
# (GPT-2 tarzı) arkasına mı (Kumru sözlüğünde daha sık) eklendiği sonucu değiştirdiği için ikisi de ölçülür.
WORD_SPLIT = {
    "bosluk_onde": r"\p{N}| ?\p{L}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "bosluk_arkada": r"\p{N}|\p{L}+ ?| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
# Çok kelimeli token: boşlukla ayrılmış en az iki harf dizisi (yalnız boşluktan oluşan tokenlar sayılmaz).
TWO_WORDS = regex.compile(r"\p{L}+\s+\p{L}+")


def flores(lang: str) -> list[str]:
    lines = []
    for split in ("dev", "devtest"):
        path = hf_hub_download(FLORES[0], f"{split}/{lang}.jsonl", repo_type="dataset", revision=FLORES[1])
        with open(path, encoding="utf-8") as fh:
            rows = sorted((json.loads(line) for line in fh), key=lambda r: r["id"])
        lines += [r["text"] for r in rows]
    return lines


def count(tok: Tokenizer, texts: list[str]) -> int:
    return sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "kumru_pretok.json")
    args = parser.parse_args()
    path = hf_hub_download(REPO, "tokenizer.json", revision=REV)
    tok = Tokenizer.from_file(path)
    split_toks = {}
    for name, rx in WORD_SPLIT.items():
        t = Tokenizer.from_file(path)
        t.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(rx), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        split_toks[name] = t
    with open(path, encoding="utf-8") as fh:
        regex = json.load(fh)["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    vocab = tok.get_vocab()

    def multi_word(token: str) -> bool:
        return bool(TWO_WORDS.search(tok.decoder.decode([token])))

    inner_space = [t for t in vocab if "Ġ" in t[1:-1]]
    multi_vocab = [t for t in inner_space if multi_word(t)]
    decoded = [tok.decoder.decode([t]) for t in vocab]
    letter_with = {
        ch: sorted(s for s in decoded if ch in s and any(c.isalpha() for c in s)) for ch in ("\n", "\t")
    }
    tr, en = flores("tur_Latn"), flores("eng_Latn")
    encoded_tr = tok.encode_batch(tr, add_special_tokens=False)
    multi_tr = sum(1 for e in encoded_tr for t in e.tokens if "Ġ" in t[1:-1] and multi_word(t))
    total_tr = sum(len(e.ids) for e in encoded_tr)
    result = {
        "repo": f"{REPO}@{REV}",
        "flores": f"{FLORES[0]}@{FLORES[1]}",
        "regex": regex,
        "sozluk": len(vocab),
        "ortasinda_bosluk_olan_token": len(inner_space),
        "cok_kelimeli_token": len(multi_vocab),
        "harf_ve_yeni_satir_iceren_token": len(letter_with["\n"]),
        "harf_ve_tab_iceren_token": len(letter_with["\t"]),
        "harf_ve_yeni_satir_ornek": letter_with["\n"][:10],
        "harf_ve_tab_ornek": letter_with["\t"][:10],
        "kumru_tr_token": total_tr,
        "kumru_en_token": count(tok, en),
        "tr_tokenlarin_cok_kelimeli_orani": multi_tr / total_tr,
        "cumle": len(tr),
    }
    result["kumru_tr_en"] = result["kumru_tr_token"] / result["kumru_en_token"]
    for name, t in split_toks.items():
        result[f"{name}_tr_token"] = count(t, tr)
        result[f"{name}_en_token"] = count(t, en)
        result[f"{name}_tr_en"] = result[f"{name}_tr_token"] / result[f"{name}_en_token"]
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for key, value in result.items():
        if key != "regex":
            print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
