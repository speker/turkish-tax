"""GGUF metadata'sındaki imatrix yollarını dosya adına indirir; tensörlere dokunmaz.

Kullanım: python clean_gguf_meta.py GİRDİ ÇIKTI [--name AD]
llama-quantize `quantize.imatrix.file` ve `quantize.imatrix.dataset`'e, llama-imatrix `imatrix.datasets`'e
komut satırındaki yolu yazar. Yalnız bu üç anahtar dosya adına iner; --name verilirse `general.name` değişir.
Başka bir anahtarda mutlak yol varsa betik durur. Çıktı yeniden okunur ve girdiyle karşılaştırılır:
anahtarlar, tipler, değerler (beyan edilen değişiklikler dışında), hizalama, tensör adı, türü, şekli ve
verisi aynı olmalı; hiçbir dizede (sözlük dahil) ev dizini geçmemeli. Sonuç JSON satırı: değişen anahtarlar,
dosya boyutu, dosyanın ve tensör verisinin SHA-256'sı. Denetim geçmezse çıkış kodu 1.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import gguf

PATH_KEYS = {"quantize.imatrix.file", "quantize.imatrix.dataset", "imatrix.datasets"}
VOCAB_KEYS = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges"}
ABSOLUTE = re.compile(r"^(/|[A-Za-z]:[\\/]|\\\\)")
ARCH = gguf.Keys.General.ARCHITECTURE
ARRAY = gguf.GGUFValueType.ARRAY


def strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def strip_path(value):
    if isinstance(value, list):
        return [strip_path(v) for v in value]
    if not (isinstance(value, str) and ABSOLUTE.match(value)):
        return value
    return (PurePosixPath if value.startswith("/") else PureWindowsPath)(value).name


def fields(reader: gguf.GGUFReader) -> dict[str, tuple[list[int], object]]:
    return {
        f.name: ([int(t) for t in f.types], f.contents())
        for f in reader.fields.values()
        if not f.name.startswith("GGUF.")
    }


def tensor_digest(reader: gguf.GGUFReader) -> tuple[str, list[tuple]]:
    h = hashlib.sha256()
    layout = []
    for t in reader.tensors:
        layout.append((t.name, int(t.tensor_type), [int(x) for x in t.shape], int(t.n_bytes)))
        h.update(t.name.encode())
        h.update(memoryview(t.data).cast("B"))
    return h.hexdigest(), layout


def file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--name")
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.input)
    before = fields(reader)
    expected, changed = {}, []
    for key, (types, value) in before.items():
        new = value
        if key in PATH_KEYS:
            new = strip_path(value)
        elif key == gguf.Keys.General.NAME and args.name:
            new = args.name
        elif key not in VOCAB_KEYS and any(ABSOLUTE.match(s) for s in strings(value)):
            raise SystemExit(f"beklenmeyen mutlak yol: {key}")
        expected[key] = (types, new)
        if new != value:
            changed.append({"key": key, "new": new})

    arch = before.get(ARCH)
    writer = gguf.GGUFWriter(args.output, arch[1] if arch else "")
    if not arch:
        writer.remove_key(ARCH)
    writer.data_alignment = reader.alignment
    for f in reader.fields.values():
        if f.name == ARCH or f.name.startswith("GGUF."):
            continue
        sub_type = f.types[-1] if f.types[0] == ARRAY else None
        writer.add_key_value(f.name, expected[f.name][1], f.types[0], sub_type=sub_type)
    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()

    out = gguf.GGUFReader(args.output)
    after = fields(out)
    digest_in, layout_in = tensor_digest(reader)
    digest_out, layout_out = tensor_digest(out)
    home = os.path.expanduser("~")
    checks = {
        "metadata_as_expected": after == expected,
        "alignment_same": out.alignment == reader.alignment,
        "tensors_identical": digest_in == digest_out and layout_in == layout_out,
        "no_home_dir": not any(home in s for _, v in after.values() for s in strings(v)),
    }
    print(
        json.dumps(
            {
                "file": PurePosixPath(args.output).name,
                "changed": changed,
                "size": Path(args.output).stat().st_size,
                "file_sha256": file_digest(args.output),
                "tensors": len(layout_out),
                "tensor_sha256": digest_out,
                **checks,
            },
            ensure_ascii=False,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
