"""Veri dosyalarında kişisel veri taraması: TCKN, TR IBAN, +90 telefon, e-posta, IP.

Kullanım: python tools/pii_scan.py DOSYA_YA_DA_KLASÖR... [--limit N]
Metin dosyaları satır satır, .parquet dosyaları (pyarrow kuruluysa) metin kolonları satır satır taranır.
TCKN checksum'la, IBAN MOD 97-10 ile doğrulanır. RFC 2606 alan adları ve RFC 5737/3849 belge IP'leri,
loopback ve unspecified adresler sayılmaz. Bulunan değer hiçbir zaman yazdırılmaz; yalnız yer ve tür.
Çıkış kodu: 0 temiz, 1 bulgu var, 2 kullanım hatası.
"""

import argparse
import ipaddress
import re
import sys
from collections import Counter
from pathlib import Path

# Dizin taramasında atlanır: paket hash'i ve boyutu taşır, rakam dizileri TCKN ve telefon kalıbına benzer
LOCKFILES = {"uv.lock"}
TCKN = re.compile(r"(?<![\d.])[1-9]\d{10}(?!\d)")
IBAN = re.compile(r"(?<![A-Za-z0-9])TR\d{2}(?:\s?\d){22}(?!\d)", re.IGNORECASE)
PHONE = re.compile(
    r"(?<![\d+.])(?:(?:\+90|0090)[\s.-]?\(?[2-5]\d{2}\)?|0\s?\(?[2-5]\d{2}\)?|\(?5\d{2}\)?(?=[\s.-]))"
    r"[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)"
)
EMAIL = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6 = re.compile(r"(?<![\w:])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![\w:])")
RESERVED_DOMAINS = re.compile(r"(^|\.)(example\.(com|net|org)|test|example|invalid|localhost)$", re.I)
# 0.0.0.0/8 ana makine adresi olamaz; llama.cpp log zaman damgaları ("0.28.254.199") bu biçimde.
DOC_NETS = [
    ipaddress.ip_network(n)
    for n in ("0.0.0.0/8", "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "2001:db8::/32")
]


def valid_tckn(s: str) -> bool:
    d = [int(c) for c in s]
    return d[9] == ((sum(d[0:9:2]) * 7 - sum(d[1:8:2])) % 10) and d[10] == sum(d[:10]) % 10


def valid_iban(s: str) -> bool:
    s = re.sub(r"\s", "", s).upper()
    rearranged = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def valid_ip(s: str) -> bool:
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_unspecified or any(ip in net for net in DOC_NETS))


def findings(text: str) -> list[str]:
    kinds = []
    kinds += ["TCKN" for m in TCKN.finditer(text) if valid_tckn(m.group(0))]
    kinds += ["IBAN" for m in IBAN.finditer(text) if valid_iban(m.group(0))]
    kinds += ["TELEFON" for _ in PHONE.finditer(text)]
    kinds += ["EPOSTA" for m in EMAIL.finditer(text) if not RESERVED_DOMAINS.search(m.group(1))]
    kinds += ["IP" for m in IPV4.finditer(text) if valid_ip(m.group(0))]
    kinds += ["IP" for m in IPV6.finditer(text) if m.group(0).count(":") >= 2 and valid_ip(m.group(0))]
    return kinds


def text_rows(path: Path):
    data = path.read_bytes()
    if b"\0" in data[:8192] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16", errors="replace")
    else:
        text = data.decode("utf-8-sig", errors="replace")
    for no, line in enumerate(text.splitlines(), 1):
        yield str(no), line


def parquet_rows(path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = [f.name for f in table.schema if str(f.type) in ("string", "large_string")]
    for name in columns:
        for i, value in enumerate(table.column(name).to_pylist()):
            if value:
                yield f"{i}:{name}", value


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Kişisel veri taraması")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--limit", type=int, default=200, help="yazdırılacak en fazla bulgu satırı")
    args = parser.parse_args()

    files = []
    for target in map(Path, args.paths):
        if target.is_dir():
            files += sorted(
                p
                for p in target.rglob("*")
                if p.is_file() and ".git" not in p.parts and p.name not in LOCKFILES
            )
        elif target.is_file():
            files.append(target)
        else:
            parser.error(f"yok: {target}")

    total, printed = Counter(), 0
    for path in files:
        rows = parquet_rows(path) if path.suffix.lower() == ".parquet" else text_rows(path)
        for where, text in rows:
            for kind in findings(text):
                total[kind] += 1
                if printed < args.limit:
                    print(f"{path.as_posix()}:{where} {kind}")
                    printed += 1
    summary = ", ".join(f"{k} {v}" for k, v in sorted(total.items())) or "bulgu yok"
    print(f"pii_scan: {len(files)} dosya; {summary}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
