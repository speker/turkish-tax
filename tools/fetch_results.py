"""Sunucudaki sonuç dosyalarını repoya kopyalar ve kişisel/iç bilgiyi maskeler.

Kullanım: python tools/fetch_results.py UZAK_DIZIN YEREL_DIZIN [--host h200] [--glob '*.log' ...]
    [--depth 1]
Yalnız kalıba uyan, 50 MB'tan küçük metin dosyaları sunucuda seçilip aktarılır; model ve ikili dosya
taşınmaz. Ev dizini yolu "~", host adı "<host>", IPv4 adresleri "<IP>" olur. Kopyalananlar pii_scan ile
taranır; bulgu varsa çıkış kodu sıfırdan farklı.
"""

import argparse
import io
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
TEXT_SUFFIXES = (".log", ".txt", ".tsv", ".csv", ".json", ".jsonl")
MAX_BYTES = "50M"


def ssh(host: str, cmd: str) -> bytes:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, check=True, timeout=600
    ).stdout


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_dir")
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("--host", default="h200")
    parser.add_argument("--glob", nargs="*", default=[f"*{s}" for s in TEXT_SUFFIXES])
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()
    home = ssh(args.host, "echo $HOME").decode().strip()
    hostname = ssh(args.host, "hostname").decode().strip()
    names = " -o ".join(f"-name {shlex.quote(g)}" for g in args.glob)
    cmd = (
        f"cd {shlex.quote(args.remote_dir)} && find . -maxdepth {args.depth} -type f \\( {names} \\) "
        f"-size -{MAX_BYTES} -print0 | tar --null -cf - -T -"
    )
    args.local_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    with tarfile.open(fileobj=io.BytesIO(ssh(args.host, cmd))) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(TEXT_SUFFIXES):
                continue
            text = archive.extractfile(member).read().decode("utf-8", errors="replace")
            text = text.replace(home, "~").replace(hostname, "<host>")
            # llama.cpp günlük zaman damgaları (0.28.254.199) IPv4'e benzer; 0. ve 1. ile başlayanlar korunur
            text = IPV4.sub(lambda m: m[0] if m[0].startswith(("0.", "1.")) else "<IP>", text)
            dst = args.local_dir / Path(member.name).relative_to(".")
            dst.parent.mkdir(parents=True, exist_ok=True)
            # bayt olarak: write_text Windows'ta \n'yi \r\n'ye çevirir, \r\n satırlar \r\r\n olurdu
            dst.write_bytes(text.encode("utf-8"))
            copied += 1
    print(f"{copied} dosya kopyalandı ve maskelendi: {args.local_dir}")
    scan = subprocess.run([sys.executable, str(Path(__file__).with_name("pii_scan.py")), str(args.local_dir)])
    return scan.returncode


if __name__ == "__main__":
    sys.exit(main())
