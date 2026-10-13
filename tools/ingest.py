"""Load sources into the SQLite corpus.

    python -m tools.ingest                          # every downloaded default source in data/raw/
    python -m tools.ingest --quran-only
    python -m tools.ingest data/sample/fixture.jsonl   # a JSONL file of passages (see docs/DATA_SCHEMA.md)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from app.config import get_settings
import re

from app.refs import hadith_ref, quran_ref, tafsir_ref
from app.store import Passage, Store
from app.surahs import SURAHS
from tools.sources import Source, select

QURAN_VERSES = 6236
# Tafsir commentary on one verse runs to tens of thousands of characters; it is split into parts of about this
# size so that search finds the part about the question and the model is shown that part.
TAFSIR_CHUNK = 1500
_EDITOR_NOTE = re.compile(r"\[\[.*?\]\]", re.DOTALL)  # "[[في أ: ...]]": manuscript variants in the Arabic edition
_SENTENCE_END = re.compile(r"(?<=[.!?؟।])\s+")


def _alt_number(n: float | int | str) -> str:
    """Abd al-Baqi style: 1907.01 -> 1907a, "8.02" -> 8b, 22 -> 22."""
    hundredths = round(float(n) * 100)
    whole, frac = divmod(hundredths, 100)
    return f"{whole}{chr(ord('a') + frac - 1)}" if 0 < frac <= 26 else str(whole)


def quran_passages(src: Source, data: dict) -> Iterator[Passage]:
    verses = data["quran"]
    if len(verses) != QURAN_VERSES:
        raise ValueError(f"{src.source_id}: expected {QURAN_VERSES} verses, got {len(verses)}")
    for v in verses:
        name, meaning, _ = SURAHS[v["chapter"]]
        meta = {"surah": v["chapter"], "ayah": v["verse"], "surah_name": name, "surah_meaning": meaning}
        yield Passage(quran_ref(v["chapter"], v["verse"]), "quran", src.lang, src.edition, v["text"].strip(),
                      src.source_id, src.allow_training, meta)


def chunk_text(text: str, size: int = TAFSIR_CHUNK) -> list[str]:
    """Split on paragraphs, then sentences, into parts of at most about `size` characters."""
    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        pieces += [para] if len(para) <= size else [s for s in _SENTENCE_END.split(para) if s.strip()]
    chunks: list[str] = []
    for piece in pieces:
        while len(piece) > 2 * size:  # a single run-on sentence
            cut = piece.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if chunks and len(chunks[-1]) + len(piece) + 1 <= size:
            chunks[-1] += ("\n" if "\n" not in piece else "") + piece
        else:
            chunks.append(piece)
    return chunks


def tafsir_passages(src: Source, files: dict[int, list[dict]]) -> Iterator[Passage]:
    """One passage per part of the commentary. Upstream repeats a group's commentary on every verse in the group;
    it is kept once, under the group's first verse, with the last verse in meta."""
    for surah in range(1, 115):
        entries = sorted(files[surah], key=lambda e: int(e["ayah"]))
        groups: list[list] = []  # [first ayah, last ayah, text]
        for e in entries:
            text = _EDITOR_NOTE.sub("", e["text"]).strip()
            if not text:
                continue
            if groups and groups[-1][2] == text:
                groups[-1][1] = int(e["ayah"])
            else:
                groups.append([int(e["ayah"]), int(e["ayah"]), text])
        name, meaning, _ = SURAHS[surah]
        for first, last, text in groups:
            parts = chunk_text(text)
            for i, part in enumerate(parts, 1):
                meta = {"surah": surah, "ayah": first, "ayah_to": last, "part": i, "parts": len(parts),
                        "surah_name": name, "surah_meaning": meaning}
                yield Passage(tafsir_ref(src.collection, surah, first, i), src.collection, src.lang, src.edition,
                              part, src.source_id, src.allow_training, meta)


def hadith_passages(src: Source, data: dict) -> Iterator[Passage]:
    sections = data["metadata"].get("sections", {})
    for h in data["hadiths"]:
        text = (h.get("text") or "").strip()
        if not text:
            continue  # a few numbers have no text in some editions
        meta = {
            "number": h["hadithnumber"],
            "book": h.get("reference", {}).get("book"),
            "in_book": h.get("reference", {}).get("hadith"),
            "chapter_title": sections.get(str(h.get("reference", {}).get("book")), ""),
            "grades": h.get("grades") or [],
        }
        if src.collection == "muslim" and h.get("arabicnumber"):
            meta["alt_number"] = _alt_number(h["arabicnumber"])
        yield Passage(hadith_ref(src.collection, h["hadithnumber"]), src.collection, src.lang, src.edition,
                      text, src.source_id, src.allow_training, meta)


def jsonl_passages(path: Path) -> Iterator[Passage]:
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        missing = {"ref", "collection", "lang", "text"} - r.keys()
        if missing:
            raise ValueError(f"{path}:{n}: missing fields {sorted(missing)}")
        yield Passage(r["ref"], r["collection"], r["lang"], r.get("edition", path.stem), r["text"],
                      r.get("source_id", path.stem), bool(r.get("allow_training", False)), r.get("meta", {}))


def _search_only(p: Passage) -> Passage:
    p.meta["search_only"] = True
    return p


def ingest_source(store: Store, src: Source, raw_dir: Path) -> int:
    path = raw_dir / src.filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m tools.fetch` first")
    if src.kind == "tafsir":
        rows = tafsir_passages(src, {n: json.loads((path / f"{n}.json").read_text(encoding="utf-8"))
                                     for n in range(1, 115)})
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = quran_passages(src, data) if src.kind == "quran" else hadith_passages(src, data)
    if not src.display:
        rows = (_search_only(p) for p in rows)
    n = store.upsert(rows)
    store.record_source(src.source_id, src.info())
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="*", type=Path, help="JSONL passage files to ingest instead of data/raw")
    ap.add_argument("--sources", nargs="*", help="source ids (default: all default sources)")
    ap.add_argument("--quran-only", action="store_true")
    ap.add_argument("--db", type=Path, help="override DB_PATH")
    args = ap.parse_args(argv)

    settings = get_settings()
    store = Store(args.db or settings.db_path)
    if args.jsonl:
        for path in args.jsonl:
            print(f"  {path}: {store.upsert(jsonl_passages(path))} passages")
    else:
        missing = 0
        for src in select(args.sources, args.quran_only):
            try:
                print(f"  {src.source_id:22} {ingest_source(store, src, settings.raw_dir):6d} passages")
            except FileNotFoundError as e:
                print(f"  {src.source_id:22} skipped: {e}", file=sys.stderr)
                missing += 1
        if missing:
            print(f"{missing} source(s) skipped", file=sys.stderr)
    print(f"corpus now holds {store.count()} passages in {store.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
