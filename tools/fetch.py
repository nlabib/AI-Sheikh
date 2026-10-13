"""Download the pinned source files into data/raw/ and verify their checksums.

    python -m tools.fetch                 # Quran, nine hadith books and four tafsir editions, ~400 MB
    python -m tools.fetch --quran-only
    python -m tools.fetch --sources quran-ar-tanzil quran-en-pickthall
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import httpx

from app.config import get_settings
from tools.sources import Source, select


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tafsir(directory: Path) -> str:
    """A tafsir is 114 files; its checksum is over all of them, in surah order."""
    h = hashlib.sha256()
    for surah in range(1, 115):
        h.update((directory / f"{surah}.json").read_bytes())
    return h.hexdigest()


def fetch_tafsir(src: Source, raw_dir: Path, force: bool = False) -> Path:
    dest = raw_dir / src.filename
    if not force and all((dest / f"{n}.json").exists() for n in range(1, 115)) and sha256_tafsir(dest) == src.sha256:
        print(f"  ok      {src.source_id:22} (cached)")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for surah in range(1, 115):
            errors = []
            for url in src.urls:
                try:
                    r = client.get(url.format(surah=surah))
                    r.raise_for_status()
                    (dest / f"{surah}.json").write_bytes(r.content)
                    break
                except httpx.HTTPError as e:
                    errors.append(f"{url.format(surah=surah)}: {e}")
            else:
                raise RuntimeError(f"could not fetch {src.source_id} surah {surah}:\n  " + "\n  ".join(errors))
    got = sha256_tafsir(dest)
    if got != src.sha256:
        raise RuntimeError(f"{src.source_id}: checksum mismatch ({got[:12]}…); upstream changed, nothing ingested")
    size = sum(f.stat().st_size for f in dest.iterdir()) / 1e6
    print(f"  fetched {src.source_id:22} {size:6.1f} MB")
    return dest


def fetch_one(src: Source, raw_dir: Path, force: bool = False) -> Path:
    if src.kind == "tafsir":
        return fetch_tafsir(src, raw_dir, force)
    dest = raw_dir / src.filename
    if dest.exists() and not force and sha256_file(dest) == src.sha256:
        print(f"  ok      {src.source_id:22} (cached)")
        return dest
    errors = []
    for url in src.urls:
        try:
            with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            got = sha256_file(tmp)
            if got != src.sha256:
                tmp.unlink()
                errors.append(f"{url}: checksum mismatch ({got[:12]}…)")
                continue
            tmp.replace(dest)
            print(f"  fetched {src.source_id:22} {dest.stat().st_size / 1e6:6.1f} MB")
            return dest
        except httpx.HTTPError as e:
            errors.append(f"{url}: {e}")
    raise RuntimeError(f"could not fetch {src.source_id}:\n  " + "\n  ".join(errors))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="*", help="source ids from tools/sources.py (default: all default sources)")
    ap.add_argument("--quran-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download even if a verified copy exists")
    args = ap.parse_args(argv)

    raw_dir = get_settings().raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for src in select(args.sources, args.quran_only):
        try:
            fetch_one(src, raw_dir, args.force)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
